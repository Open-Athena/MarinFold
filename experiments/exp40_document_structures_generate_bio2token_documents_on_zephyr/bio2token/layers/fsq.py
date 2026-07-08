# SPDX-FileCopyrightText: 2025 flagshippioneering
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from typing import List, Optional
import torch
import torch.nn as nn
from einops import pack, rearrange, unpack
from torch import Tensor, int32
from torch.nn import Module


@dataclass
class FSQConfig:
    """
    Configuration parameters for the Finite Scalar Quantization (FSQ) module.

    Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
    Code adapted from Jax version in Appendix A.1

    Attributes:
        levels (List[int]): A list specifying the number of quantization levels per bin. Each entry in the list represents
                            the granularity of quantization for a specific bin.
        num_codebooks (int): The number of codebooks to be used in the quantization process.
        d_input (Optional[int]): The dimension of the input expected by the quantizer. If not set, it defaults to the
                                 product of num_codebooks and the length of levels, defining the effective codebook dimension.
        keep_num_codebooks_dim (Optional[bool]): Determines if the num_codebooks dimension should be retained in processes
                                                 where dimensionality may normally be reduced or altered.
    """

    levels: List[int] = field(default_factory=lambda: [4, 4, 4, 4, 4, 4])
    num_codebooks: int = 1
    d_input: Optional[int] = None
    keep_num_codebooks_dim: Optional[bool] = None


# helper functions
def default(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


class FSQ(Module):
    """
    Main class for Finite Scalar Quantization (FSQ), enabling efficient data representation through quantization.

    The FSQ class implements a quantization mechanism based on the Finite Scalar Quantization process, where data
    is quantized into a finite set of levels.

    Attributes:
        config_cls (type): The associated configuration class, set to FSQConfig.
        config (FSQConfig): An instance of FSQConfig providing parameters like number of codebooks and levels per bin.
        levels (List[int]): List indicating the number of quantization levels per bin.
        codebook_dim (int): The number of bins needed per codebook.
        num_codebooks (int): Total number of codebooks used in the quantizer.
        effective_codebook_dim (int): Combined dimensions of all codebooks.
        keep_num_codebooks_dim (bool): Flag for maintaining the num_codebooks dimension where applicable.
        d_input (int): Input dimension to the quantizer, defaults to effective codebook dimension if unspecified.
        has_projections (bool): Indicates if input projection to effective codebook dimension is necessary.
        project_in (nn.Module): Layer to project input dimensions to match effective codebook dimension. Becomes Identity if dimensions already match.
        project_out (nn.Module): Layer to project output dimensions back to d_input. Becomes Identity if dimensions already match.
        codebook_size (int): Size of the codebook, calculated as the product of all levels.

    Args:
        config (FSQConfig): Configuration object specifying quantization parameters, including levels and dimensionality settings.

    Notes:
        - The class includes input and output projection layers to adapt data to the effective codebook dimension, when needed.
    """

    config_cls = FSQConfig

    def __init__(
        self,
        config: FSQConfig,
    ):
        super(FSQ, self).__init__()
        self.config = config

        # Define quantization bins based on configuration
        self.levels = self.config.levels
        _levels = torch.tensor(self.levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        # Calculate the basis for quantization operations
        _basis = torch.cumprod(torch.tensor([1] + self.levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        # Determine the number of bins per codebook and overall codebook dimensions
        self.codebook_dim = len(self.levels)
        self.num_codebooks = self.config.num_codebooks
        self.effective_codebook_dim = self.codebook_dim * self.num_codebooks

        # Manage dimensional configurations and projections
        self.keep_num_codebooks_dim = default(self.config.keep_num_codebooks_dim, self.config.num_codebooks > 1)
        assert not (self.config.num_codebooks > 1 and not self.keep_num_codebooks_dim)

        # Determine input dimensionality and manage projections
        self.d_input = default(self.config.d_input, self.effective_codebook_dim)
        self.has_projections = self.d_input != self.effective_codebook_dim
        self.project_in = nn.Linear(self.d_input, self.effective_codebook_dim) if self.has_projections else nn.Identity()
        self.project_out = nn.Linear(self.effective_codebook_dim, self.d_input) if self.has_projections else nn.Identity()

        # Calculate the total size of the codebook
        self.codebook_size = self._levels.prod().item()

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """
        Bound the input tensor `z` to specified limits based on the number of quantization levels.

        This function applies a bounding operation to `z`, tailoring the bounds according to
        whether the number of levels is even or odd. For odd levels, the values are normalized
        using a hyperbolic tangent (tanh) function to fit between a symmetric range. For even levels,
        a shift is introduced to adjust the bounds slightly asymmetrically.

        Args:
            z (Tensor): The input tensor of shape (..., d) representing data to be bounded.
            eps (float): A small epsilon value to ensure numerical stability during computations. Defaults to 1e-3.

        Returns:
            Tensor: A tensor of the same shape as `z`, with values bounded according to the specified rules.

        Notes:
            - z is bounded by [-1, 0] for l=2, [-1, 1] for l=3, [-2, 1] for l=4, [-2, 2] for l=5, etc.
            - For odd levels:
            z_bounded = tanh(z) * (L - 1) / 2
            bounded by [-(L - 1) / 2, (L - 1) / 2]
            - For even levels:
            z_bounded = [tanh(z + 1 / (L - 1)) * (L - 1) - 1] / 2
            bounded by [-L / 2, (L - 2) / 2]
        """
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()  # Why do we have a shift for even levels?
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """
        Quantize the input tensor `z` into discrete levels and return the quantized output.

        This function applies a quantization process to `z`, fitting the values into specified discrete levels.
        The quantization allows the original values to be approximated in a continuous domain for backpropagation,
        even though the actual rounding during the quantization process does not support gradients.

        Args:
            z (Tensor): The input tensor to be quantized, containing continuous values that will be mapped to discrete levels.

        Returns:
            Tensor: A quantized tensor `zhat`, of the same shape as `z`, with values discretized and renormalized.

        Notes:
            - The rounding operation used in quantization is not differentiable, meaning that the gradient flow in backpropagation
            relies on a continuous approximation. This differentiable approximation takes place in the range established for `z`.
            uantizes z, returns quantized zhat, same shape as z.
            - For example, the code is discretized into {-2, -1, 0, 1} for L=4. But the gradient is associated with the continous value of z,
            which can be anything between [-2, 1].
            - After quantization, everything is renormalized to [-1, 1], i.e. {-1, -0.5, 0, 0.5} for L=4.
        """
        z = self.bound(z)
        quantized = z + (z.round() - z).detach()
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """
        Convert quantized codes to indices in the codebook.

        This method transforms the quantized tensor `zhat` into indices that map to
        positions in a quantization codebook, facilitating efficient lookups and retrievals
        from quantized representations.

        Args:
            zhat (Tensor): A tensor representing quantized values, expected to have a last dimension size
                        equal to `codebook_dim`.

        Returns:
            Tensor: A tensor of indices, where each index corresponds to a position in the codebook.

        Notes:
            - The conversion utilizes scaling and shifting to appropriately map the quantized codes
            into discrete indices, supported by basis vector scaling.
            - The `codebook_dim` ensures that only the appropriate representations are converted
            for each quantization level.
        """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(self, indices: Tensor, project_out=True) -> Tensor:
        """
        Convert indices from the codebook back to quantized codes.

        This method performs the inverse of `codes_to_indices`, reconstructing quantized
        codes from given indices in the codebook. The reconstructed codes can optionally
        be projected to a desired output dimension.

        Args:
            indices (Tensor): A tensor of indices corresponding to positions in the quantization codebook.
            project_out (bool): A flag indicating whether to apply output dimensionality projection
                                to the reconstructed codes. Defaults to True.

        Returns:
            Tensor: A tensor of quantized codes reconstructed from the indices, optionally projected
                    to a different dimension.

        Notes:
            - The reconstruction respects the codebook dimensionality and optionally projects
            the results using a configured linear projection.
            - Handles image or video data by appropriately rearranging dimensions to support
            batch processing and spatial formats.
            - The method applies inverse scaling and shifting to revert the indices to original code values.
        """
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def forward(self, z: Tensor, with_hidden_codes: bool = False) -> Tensor:
        """
        Perform the forward pass to quantize input tensors and return the quantized output.

        This method processes the input tensor `z`, applying dimensional transformations, quantization,
        and codebook indexing. It optionally supports handling multi-dimensional image or video data,
        reformulating them for batch processing, and returns the quantized outputs along with indices,
        and optionally the hidden codes.

        Args:
            z (Tensor): The input tensor with dimensions corresponding to batch, sequence (or spatial),
                        and feature dimensions.
            with_hidden_codes (bool): Flag indicating whether to return the latent quantized codes in addition
                                    to the final output and indices. Defaults to False.

        Returns:
            Tensor: The quantized output after passing through the quantization and projection layers.
            Tensor: Indices representing positions in the codebook corresponding to the quantized values.
            Tensor (optional): The latent quantized codes, provided if `with_hidden_codes` is True.

        Notes:
            - Handles input reshaping for image or video data to standardize processing.
            - Ensures import dimensions match expected input dimensions, adapting as necessary.
            - Uses Einstein notation for tensor manipulation, where:
            b - batch size
            n - sequence length or flattened spatial dimensions
            d - feature dimension or log2(codebook size)
            c - number of codebook dimensions
        """
        # Determine if the input tensor represents an image or video by checking the number of dimensions
        is_img_or_video = z.ndim >= 4

        # Standardize image or video to (batch, seq, dimension) format
        if is_img_or_video:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack([z], "b * d")

        # Check that the feature dimension matches the expected input dimension
        assert z.shape[-1] == self.d_input, f"expected dimension of {self.d_input} but found dimension of {z.shape[-1]}"

        # Project the input to match the effective codebook dimension
        z = self.project_in(z)

        # Rearrange tensor for codebook quantization
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        # Quantize the rearranged tensor
        codes = self.quantize(z)
        # Convert the quantized codes to indices for the codebook
        indices = self.codes_to_indices(codes)

        # Rearrange quantized codes back to a combined dimension
        codes = rearrange(codes, "b n c d -> b n (c d)")

        # Project the quantized codes back to the original input dimension
        out = self.project_out(codes)

        # If processing image or video data, unpack the outputs to the original spatial format
        if is_img_or_video:
            out = unpack(out, ps, "b * d")[0]
            out = rearrange(out, "b ... d -> b d ...")

            indices = unpack(indices, ps, "b * c")[0]

        # Remove the singleton codebook dimension from indices if not needed
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        # Return output and indices; optionally return latent quantized codes if requested
        if with_hidden_codes:
            return out, indices, codes
        else:
            return out, indices
