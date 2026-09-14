# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint overlay construction for path-based inference loaders."""

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from marinfold.inference._config import needs_rope_repair, read_config, repair_rope
from marinfold.inference._tokenizer import load_tokenizer, tokenizer_source_path

LEVANTER_TOKEN_EMBEDDING_KEY = "token_embeddings.weight"
HF_TOKEN_EMBEDDING_KEY = "model.embed_tokens.weight"
POSITION_TOKEN_PATTERN = re.compile(r"^<p(\d+)>$")


@dataclass(frozen=True)
class PositionTokenSpan:
    """Contiguous residue-position token range in the tokenizer vocabulary."""

    start_token_id: int
    num_tokens: int
    base: float = 10_000.0

    @classmethod
    def from_config(cls, config: dict) -> "PositionTokenSpan | None":
        spec = config.get("position_embedding")
        if not isinstance(spec, dict):
            return None
        if "start_token_id" not in spec or "num_tokens" not in spec:
            return None
        return cls(
            start_token_id=int(spec["start_token_id"]),
            num_tokens=int(spec["num_tokens"]),
            base=float(spec.get("base", 10_000.0)),
        )

    @classmethod
    def from_tokenizer(cls, model_path: Path) -> "PositionTokenSpan | None":
        tokenizer = load_tokenizer(model_path)
        positions: list[tuple[int, int]] = []
        for token, token_id in tokenizer.get_vocab().items():
            match = POSITION_TOKEN_PATTERN.match(token)
            if match is not None:
                positions.append((int(match.group(1)), int(token_id)))
        if not positions:
            return None

        positions.sort()
        actual_positions = [position for position, _ in positions]
        actual_ids = [token_id for _, token_id in positions]
        expected_positions = list(range(len(positions)))
        expected_ids = list(range(actual_ids[0], actual_ids[0] + len(actual_ids)))
        if actual_positions != expected_positions or actual_ids != expected_ids:
            return None
        return cls(start_token_id=actual_ids[0], num_tokens=len(actual_ids))

    def with_env_overrides(self) -> "PositionTokenSpan":
        return PositionTokenSpan(
            start_token_id=int(
                os.environ.get(
                    "MARINFOLD_FIXED_RESIDUE_POSITION_START_TOKEN_ID",
                    self.start_token_id,
                )
            ),
            num_tokens=int(
                os.environ.get(
                    "MARINFOLD_FIXED_RESIDUE_POSITION_NUM_TOKENS",
                    self.num_tokens,
                )
            ),
            base=float(
                os.environ.get("MARINFOLD_FIXED_RESIDUE_POSITION_BASE", self.base)
            ),
        )

    @classmethod
    def from_env(cls) -> "PositionTokenSpan | None":
        start = os.environ.get("MARINFOLD_FIXED_RESIDUE_POSITION_START_TOKEN_ID")
        num_tokens = os.environ.get("MARINFOLD_FIXED_RESIDUE_POSITION_NUM_TOKENS")
        if start is None or num_tokens is None:
            return None
        return cls(
            start_token_id=int(start),
            num_tokens=int(num_tokens),
            base=float(os.environ.get("MARINFOLD_FIXED_RESIDUE_POSITION_BASE", 10_000.0)),
        )

    def validate(self, *, vocab_size: int) -> None:
        stop = self.start_token_id + self.num_tokens
        if self.start_token_id < 0 or self.num_tokens <= 0 or stop > vocab_size:
            raise ValueError(
                "invalid fixed residue-position span: "
                f"start={self.start_token_id} num_tokens={self.num_tokens} "
                f"vocab_size={vocab_size}"
            )


def normalize_embedding_repair_mode(mode: str | None) -> str:
    value = mode or os.environ.get("MARINFOLD_FIXED_RESIDUE_POSITION_EMBEDDINGS", "auto")
    if value not in {"auto", "force", "off"}:
        raise ValueError(
            "fixed residue-position embedding repair mode must be one of "
            f"'auto', 'force', or 'off'; got {value!r}"
        )
    return value


def fixed_rope_rows(num_positions: int, embed_dim: int, *, base: float) -> np.ndarray:
    """Return deterministic residue-position rows matching exp157 training."""

    if embed_dim % 2 != 0:
        raise ValueError(
            f"fixed residue-position embeddings require an even dimension, got {embed_dim}"
        )
    positions = np.arange(num_positions, dtype=np.float32)[:, None]
    channels = np.arange(embed_dim // 2, dtype=np.float32)
    inv_freq = base ** (-2.0 * channels / embed_dim)
    angles = positions * inv_freq[None, :]
    return np.stack((np.sin(angles), np.cos(angles)), axis=-1).reshape(
        num_positions,
        embed_dim,
    )


@dataclass
class CheckpointOverlayBuilder:
    """Build a temporary checkpoint directory only when a loader needs one."""

    model_path: Path
    fixed_residue_position_embeddings: str | None = None

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        self.tokenizer_path = Path(tokenizer_source_path(self.model_path))
        self.raw_config = read_config(self.model_path)
        self.repaired_config = repair_rope(self.raw_config)
        self.repair_config = needs_rope_repair(self.raw_config)
        self.embedding_repair_mode = normalize_embedding_repair_mode(
            self.fixed_residue_position_embeddings
        )
        self.repair_embeddings = self.needs_embedding_repair()
        self.overlay_path: Path | None = None

    def build(self) -> str:
        if not self.needs_overlay:
            return str(self.model_path)

        self.overlay_path = self.tokenizer_path
        if self.overlay_path == self.model_path:
            self.overlay_path = Path(tempfile.mkdtemp(prefix="marinfold-checkpoint-"))
        self.add_checkpoint_files()
        if self.repair_config:
            self.write_config()
        if self.repair_embeddings:
            self.write_token_embeddings()
        return str(self.overlay_path)

    @property
    def needs_overlay(self) -> bool:
        return (
            self.tokenizer_path != self.model_path
            or self.repair_config
            or self.repair_embeddings
        )

    @property
    def config(self) -> dict:
        return self.repaired_config if self.repair_config else self.raw_config

    def needs_embedding_repair(self) -> bool:
        if self.embedding_repair_mode == "off":
            return False
        index_path = self.model_path / "model.safetensors.index.json"
        if not index_path.exists():
            return False
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        return (
            LEVANTER_TOKEN_EMBEDDING_KEY in weight_map
            and HF_TOKEN_EMBEDDING_KEY not in weight_map
        )

    def add_checkpoint_files(self) -> None:
        assert self.overlay_path is not None
        for model_entry in self.model_path.iterdir():
            destination = self.overlay_path / model_entry.name
            if destination.exists() or destination.is_symlink():
                continue
            destination.symlink_to(
                model_entry.resolve(), target_is_directory=model_entry.is_dir()
            )

    def write_config(self) -> None:
        assert self.overlay_path is not None
        destination = self.overlay_path / "config.json"
        destination.unlink(missing_ok=True)
        destination.write_text(json.dumps(self.config, indent=2) + "\n")

    def write_token_embeddings(self) -> None:
        """Write a standard HF input embedding table into the overlay."""

        assert self.overlay_path is not None
        from safetensors.numpy import load_file, save_file

        index_path = self.overlay_path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        source_shard = weight_map.get(LEVANTER_TOKEN_EMBEDDING_KEY)
        if source_shard is None:
            return

        shard_path = self.overlay_path / source_shard
        tensors = load_file(shard_path)
        learned = tensors.pop(LEVANTER_TOKEN_EMBEDDING_KEY)
        tensors[HF_TOKEN_EMBEDDING_KEY] = self.materialize_embedding_table(learned)

        if shard_path.is_symlink():
            shard_path.unlink()
        save_file(tensors, shard_path)
        weight_map.pop(LEVANTER_TOKEN_EMBEDDING_KEY)
        weight_map[HF_TOKEN_EMBEDDING_KEY] = source_shard
        if index_path.is_symlink():
            index_path.unlink()
        index_path.write_text(json.dumps(index, indent=2) + "\n")

    def materialize_embedding_table(self, learned: np.ndarray) -> np.ndarray:
        vocab_size = int(self.config["vocab_size"])
        configured_span = PositionTokenSpan.from_config(self.config)
        if configured_span is not None:
            configured_span = configured_span.with_env_overrides()
        span = configured_span or PositionTokenSpan.from_tokenizer(self.model_path)
        if span is not None:
            span = span.with_env_overrides()
        else:
            span = PositionTokenSpan.from_env()

        if span is None:
            if learned.shape[0] != vocab_size:
                raise ValueError(
                    "compact token_embeddings.weight needs a config-declared or "
                    "tokenizer-inferable residue-position span; got "
                    f"{learned.shape} for vocab_size={vocab_size}"
                )
            return learned

        if learned.shape[0] < vocab_size:
            span = PositionTokenSpan(
                start_token_id=span.start_token_id,
                num_tokens=vocab_size - learned.shape[0],
                base=span.base,
            )
        span.validate(vocab_size=vocab_size)
        fixed = fixed_rope_rows(
            span.num_tokens,
            learned.shape[1],
            base=span.base,
        ).astype(learned.dtype)

        if learned.shape[0] == vocab_size and (
            configured_span is not None or self.embedding_repair_mode == "force"
        ):
            full = learned.copy()
            full[span.start_token_id : span.start_token_id + span.num_tokens] += fixed
            return full
        if learned.shape[0] == vocab_size:
            return learned
        if learned.shape[0] == vocab_size - span.num_tokens:
            stop = span.start_token_id + span.num_tokens
            full = np.empty((vocab_size, learned.shape[1]), dtype=learned.dtype)
            full[: span.start_token_id] = learned[: span.start_token_id]
            full[span.start_token_id : stop] = fixed
            full[stop:] = learned[span.start_token_id :]
            return full
        raise ValueError(
            "unexpected token_embeddings.weight shape for fixed residue-position "
            f"checkpoint: {learned.shape}; vocab_size={vocab_size}, "
            f"num_tokens={span.num_tokens}"
        )


def model_source_path(
    model_path: Path,
    *,
    fixed_residue_position_embeddings: str | None = None,
) -> str:
    """Return a checkpoint directory a path-based loader can read as-is.

    Path-based consumers such as vLLM and MLX load model files, config, and
    tokenizer internally. This creates a symlink overlay when a checkpoint needs
    small metadata or embedding-table repairs while leaving original checkpoint
    files untouched.
    """

    return CheckpointOverlayBuilder(
        Path(model_path),
        fixed_residue_position_embeddings=fixed_residue_position_embeddings,
    ).build()
