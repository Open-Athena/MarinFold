# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""bio2token encoder/decoder assembled from the pure-PyTorch Mamba stack.

The token path is: coords ``(B, L, 3)`` -> input projection (3->128) -> 4
bidirectional Mamba-1 layers -> ``norm_f`` -> FSQ quantizer -> integer
``indices`` in ``[0, 4095]`` (one token per atom). The decoder (128->3) is
used only to validate the encoder by round-trip reconstruction.

Module tree deliberately mirrors the official checkpoint so ``state_dict``
loads with no missing/unexpected keys:

    encoder.encoder.*   (MambaStack, 3->128)   quantizer.project_{in,out}
    decoder.decoder.*   (MambaStack, 128->3)

FSQ carries no learned codebook (levels [4]*6 -> 4096 codes); the only FSQ
parameters are the 128<->6 projections. FSQ logic is vendored from bio2token
under ``bio2token/`` (see that package's __init__ for attribution).
"""

import os
import urllib.request

import torch
import torch.nn as nn

from bio2token.layers.fsq import FSQ, FSQConfig
from mamba import MambaStack

# Official pretrained checkpoint (~14 MB, no Git-LFS). Downloaded on demand;
# not committed (repo policy: no large binaries in-tree).
CKPT_URL = (
    "https://raw.githubusercontent.com/flagshippioneering/bio2token/main/"
    "checkpoints/bio2token/bio2token_pretrained/"
    "epoch%3D0243-val_loss_epoch%3D0.71-best-checkpoint.ckpt"
)
FSQ_LEVELS = [4, 4, 4, 4, 4, 4]
CODEBOOK_SIZE = 4096  # prod(FSQ_LEVELS)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MambaStack(d_input=3, d_output=128, n_layer=4, bidirectional=True)
        self.quantizer = FSQ(FSQConfig(levels=FSQ_LEVELS, d_input=128))

    def forward(self, structure):
        encoding, indices = self.quantizer(self.encoder(structure))
        return encoding, indices


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = MambaStack(d_input=128, d_output=3, n_layer=6, bidirectional=True)

    def forward(self, encoding):
        return self.decoder(encoding)


class Bio2Token(nn.Module):
    """Encoder + decoder. ``tokenize`` is the production surface; ``decode``
    exists for the round-trip validation test."""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    @torch.no_grad()
    def tokenize(self, structure):
        """structure: (B, L, 3) float -> indices: (B, L) long in [0, 4095]."""
        _, indices = self.encoder(structure)
        return indices

    @torch.no_grad()
    def reconstruct(self, structure):
        encoding, indices = self.encoder(structure)
        return self.decoder(encoding), indices


def get_checkpoint(cache_dir: str | None = None) -> str:
    """Return a local path to the pretrained checkpoint, downloading if absent."""
    cache_dir = cache_dir or os.path.join(
        os.path.expanduser("~"), ".cache", "marinfold", "bio2token")
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, "bio2token_pretrained.ckpt")
    if not os.path.exists(dest):
        urllib.request.urlretrieve(CKPT_URL, dest)
    return dest


def load_bio2token(ckpt_path: str | None = None, device: str = "cpu") -> Bio2Token:
    """Build ``Bio2Token`` and load the official weights (Lightning state_dict,
    ``model.`` prefix stripped)."""
    ckpt_path = ckpt_path or get_checkpoint()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    stripped = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    model = Bio2Token()
    missing, unexpected = model.load_state_dict(stripped, strict=True)
    return model.to(device).eval()
