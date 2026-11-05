"""Qwen3.5 text-only MaxText adapter with explicit Hugging Face parameter mapping.

The upstream model registry has no dense 4B entry. Construct its public generic
configuration with the source architecture instead of altering that registry.
Every conversion is a permutation/reshape/concatenation, so the same mapping
applies to FP32 parameters and both Adam moments without changing coordinates.
"""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.core.spmd import logical_axis_rules
from maxtext.checkpoint_conversion.utils.param_mapping import (
    QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils, model_creation_utils
from maxtext.utils.vocabulary_tiling import vocab_tiling_nnx_loss


def configuration(
    source: dict,
    *,
    length: int,
    batch: int,
    dtype: str = "bfloat16",
    attention: str = "dot_product",
    tiles: int = 2,
) -> Any:
    """Build a text-only configuration directly from the frozen HF architecture."""
    expected_layers = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
        for i in range(source["num_hidden_layers"])
    ]
    if source["layer_types"] != expected_layers or not source["tie_word_embeddings"]:
        raise ValueError("Expected the experiment's tied dense 3:1 hybrid Qwen model")
    return pyconfig.initialize(
        ["exp347_tpu"],
        model_name="default",
        decoder_block="qwen3_5",
        base_emb_dim=source["hidden_size"],
        base_mlp_dim=source["intermediate_size"],
        base_num_decoder_layers=source["num_hidden_layers"],
        base_num_query_heads=source["num_attention_heads"],
        base_num_kv_heads=source["num_key_value_heads"],
        head_dim=source["head_dim"],
        vocab_size=source["vocab_size"],
        normalization_layer_epsilon=source["rms_norm_eps"],
        mlp_activations=["silu", "linear"],
        num_experts=1,
        inhomogeneous_layer_cycle_interval=4,
        gdn_conv_kernel_dim=source["linear_conv_kernel_dim"],
        gdn_key_head_dim=source["linear_key_head_dim"],
        gdn_value_head_dim=source["linear_value_head_dim"],
        gdn_num_key_heads=source["linear_num_key_heads"],
        gdn_num_value_heads=source["linear_num_value_heads"],
        gdn_chunk_size=64,
        use_qk_norm_in_gdn=True,
        rope_max_timescale=source["rope_parameters"]["rope_theta"],
        partial_rotary_factor=source["partial_rotary_factor"],
        use_mrope=False,
        enable_dropout=False,
        dropout_rate=0.0,
        logits_via_embedding=True,
        normalize_embedding_logits=False,
        dtype=dtype,
        weight_dtype="float32",
        num_vocab_tiling=tiles,
        vocab_tiling_ag_once=True,
        z_loss_multiplier=0.0,
        attention=attention,
        scan_layers=False,
        shard_mode="auto",
        max_target_length=length,
        max_prefill_predict_length=length,
        per_device_batch_size=batch / jax.device_count(),
        ici_fsdp_parallelism=-1,
        ici_data_parallelism=1,
        run_name="exp347-tpu",
        base_output_directory="/tmp/exp347-maxtext",
        dataset_type="synthetic",
        skip_jax_distributed_system=True,
    )


def abstract_model(config: Any) -> tuple[Any, nnx.Module]:
    """Create parameter shapes and shardings without allocating random 4B weights."""
    mesh = maxtext_utils.get_mesh_from_config(config)
    _, model = model_creation_utils.create_nnx_abstract_model(config, mesh=mesh)
    return mesh, model


def parameter_mapping(source: dict, config: Any) -> tuple[dict, dict]:
    """Return explicit text-only mappings, excluding the tied duplicate LM head."""
    mapping = QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING(
        source, config, scan_layers=False, hf_prefix="model"
    )
    del mapping["params-decoder-logits_dense-kernel"]
    hooks = QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN(
        source, config, scan_layers=False, saving_to_hf=False
    )
    return mapping, hooks


def convert_parameters(
    source: Mapping[str, np.ndarray],
    source_config: dict,
    config: Any,
    abstract: nnx.State,
) -> nnx.State:
    """Map all HF arrays to MaxText coordinates and require complete coverage."""
    mapping, hooks = parameter_mapping(source_config, config)
    result = []
    used = set()
    mapped = set()
    for path, variable in nnx.to_flat_state(abstract):
        key = "params-" + "-".join(map(str, path))
        hf_names = mapping[key]
        names = (hf_names,) if isinstance(hf_names, str) else hf_names
        if not isinstance(names, tuple) or not all(isinstance(n, str) for n in names):
            raise ValueError(f"Unexpected non-flat mapping for {key}")
        used.update(names)
        values = tuple(source[name] for name in names)
        value = values[0] if isinstance(hf_names, str) else values
        if key in hooks:
            value = hooks[key](value, variable.get_value().shape)
        # CPU device_put may share NumPy storage. Own the snapshot so a reference
        # PyTorch update cannot silently mutate the supposedly converted state.
        value = np.array(value, copy=True)
        if value.shape != variable.get_value().shape or value.dtype != np.float32:
            raise ValueError(
                f"Invalid shape/dtype for {key}: {value.shape}/{value.dtype}"
            )
        array = jax.make_array_from_callback(
            value.shape,
            variable.get_value().sharding,
            lambda index, value=value: value[index],
        )
        result.append((path, variable.replace(value=array)))
        mapped.add(key)
    if mapped != set(mapping) or used != set(source):
        raise ValueError(
            f"Incomplete conversion: missing MT={set(mapping) - mapped}; unused HF={set(source) - used}"
        )
    return nnx.from_flat_state(result)


def initial_rest(abstract: nnx.State, seed: int = 347) -> nnx.State:
    """Initialize unused dropout/quantization RNG metadata for deterministic training."""
    result = []
    for path, variable in nnx.to_flat_state(abstract):
        value = variable.get_value()
        if isinstance(variable, nnx.RngKey):
            array = jax.random.key(seed)
        elif isinstance(variable, nnx.RngCount):
            array = jax.numpy.zeros(value.shape, value.dtype)
        else:
            raise TypeError(f"Unexpected non-parameter model state: {path}")
        result.append(
            (path, variable.replace(value=jax.device_put(array, value.sharding)))
        )
    return nnx.from_flat_state(result)


class TiedHeadView(nnx.Module):
    """Expose the original tied head with an explicit, configurable compute dtype.

    Upstream attend_on_embedding always casts the table to BF16, including in
    FP32 mode. This wrapper uses its public vocabulary-tiling API while retaining
    the source model's head coordinates and the requested arithmetic precision.
    """

    def __init__(self, model: nnx.Module) -> None:
        self.model = model
        self.mesh = model.mesh

    def logits_from_hidden_states_for_vocab_tiling(
        self, hidden_states: Any, deterministic: bool, model_mode: str
    ) -> Any:
        """Normalize and project through the same embedding parameter."""
        del deterministic, model_mode
        config = self.model.config
        normalized = self.model.decoder.apply_decoder_norm(hidden_states)
        return jnp.einsum(
            "...e,ve->...v",
            normalized.astype(config.dtype),
            self.model.token_embedder.embedding[...].astype(config.dtype),
            preferred_element_type=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )


def hidden_model(
    graph: Any, params: nnx.State, rest: nnx.State, config: Any, tokens: Any, mask: Any
) -> tuple[TiedHeadView, Any]:
    """Run independent document rows and return unnormalized final hidden states."""
    model = nnx.merge(graph, params, rest, copy=True)
    positions = jnp.broadcast_to(jnp.arange(tokens.shape[1]), tokens.shape)
    with logical_axis_rules(config.logical_axis_rules):
        model(tokens, positions, mask, enable_dropout=False, model_mode="train")
    intermediate = nnx.pop(model, nnx.Intermediate)
    hidden = intermediate["decoder"]["hidden_states"].get_value()[0]
    return TiedHeadView(model), hidden


def apply_model(
    graph: Any, params: nnx.State, rest: nnx.State, config: Any, tokens: Any, mask: Any
) -> Any:
    """Return full logits for small parity probes, using the source tied head."""
    model, hidden = hidden_model(graph, params, rest, config, tokens, mask)
    with logical_axis_rules(config.logical_axis_rules):
        return model.logits_from_hidden_states_for_vocab_tiling(hidden, True, "train")


def summed_loss(
    graph: Any,
    params: nnx.State,
    rest: nnx.State,
    config: Any,
    tokens: Any,
    targets: Any,
    mask: Any,
) -> Any:
    """Compute a bounded-memory sum over valid next tokens, including the prompt."""
    model, hidden = hidden_model(graph, params, rest, config, tokens, mask)
    with logical_axis_rules(config.logical_axis_rules):
        loss, _ = vocab_tiling_nnx_loss(
            model,
            hidden,
            {"targets": targets, "targets_segmentation": mask},
            config,
            True,
        )
    return loss
