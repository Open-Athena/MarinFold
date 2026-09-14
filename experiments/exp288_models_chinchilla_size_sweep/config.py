"""Configuration catalog for exp288 Chinchilla-style size sweep."""

from dataclasses import dataclass

from fray.types import ResourceConfig
from levanter.layers.attention import AttentionBackend
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config

PREFIX = "s3://marin-us-east-02a/MarinFold/exp288_models_chinchilla_size_sweep"
VERSION = "2026.09.14.1"
WANDB_GROUP = "exp288-chinchilla-size-sweep"
TOKENIZER = "eczech/contacts-v1-tokenizer-5d68a24a899f"
VOCAB_SIZE = 2845

EPOCH_PACKED_EXAMPLES = 34_092_146
GLOBAL_BATCH_SIZE = 128
EPOCH_TRAIN_STEPS = (EPOCH_PACKED_EXAMPLES + GLOBAL_BATCH_SIZE - 1) // GLOBAL_BATCH_SIZE


@dataclass(frozen=True)
class Corpus:
    """One immutable source and its completed token cache."""

    name: str
    source: str
    cache: str
    documents: int
    shards: int
    tokens: int | None = None


@dataclass(frozen=True)
class ClusterSpec:
    """One CoreWeave GPU target profile."""

    gpu_variant: str
    gpus_per_node: int
    cpu: int
    ram: str
    disk: str

    def resources(self, *, nodes: int) -> ResourceConfig:
        return ResourceConfig.with_gpu(
            self.gpu_variant,
            count=self.gpus_per_node,
            replicas=nodes,
            cpu=self.cpu,
            ram=self.ram,
            disk=self.disk,
        )


@dataclass(frozen=True)
class SizeTrial:
    """One logical model-size trial in the Chinchilla sweep."""

    trial_id: str
    label: str
    model: Qwen3Config
    nodes: int

    @property
    def run_id(self) -> str:
        return f"contacts-v1-exp288-chinchilla-{self.trial_id}"


def qwen3_contacts_config(*, hidden_dim: int, num_layers: int, num_heads: int) -> Qwen3Config:
    """Build the Qwen3 contacts-v1 config shared by all size arms."""
    return Qwen3Config(
        max_seq_len=8192,
        hidden_dim=hidden_dim,
        intermediate_dim=4 * hidden_dim,
        num_heads=num_heads,
        num_kv_heads=8,
        num_layers=num_layers,
        rope=Llama3RotaryEmbeddingsConfig(),
        use_qk_norm=True,
        attn_backend=AttentionBackend.JAX_FLASH,
    )


def trainable_params(model: Qwen3Config) -> int:
    """Return Levanter trainable params plus Qwen3 QK norm params."""
    qk_norm_params = 2 * model.num_layers * model.actual_head_size
    return int(model.total_trainable_params(VOCAB_SIZE)) + qk_norm_params


CLUSTERS = {
    "cw-us-east-08a": ClusterSpec("GB200", 4, 32, "256g", "256g"),
    "cw-us-east-02a": ClusterSpec("H100", 8, 32, "256g", "256g"),
    "cw-rno2a": ClusterSpec("H100", 8, 32, "256g", "256g"),
}

MAX_SEQS_PER_DEVICE = {"GB200": 32, "H100": 8}


CORPORA = (
    Corpus(
        "native-afdb",
        "",
        "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/tokenized/contacts_v1/afdb/2026.08.14",
        3_963_003,
        2067,
        4_432_940_838,
    ),
    Corpus(
        "native-esm",
        "",
        "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/tokenized/contacts_v1/esm/2026.08.14",
        65_553_178,
        3338,
        70_042_923_165,
    ),
    Corpus(
        "mpnn-afdb",
        "s3://marin-us-east-02a/MarinFold/exp266/documents/*.parquet",
        "s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/tokenized/mpnn-afdb/2026.09.09.1",
        31_702_680,
        199,
        35_352_543_972,
    ),
    Corpus(
        "mpnn-esm",
        "s3://marin-us-east-02a/MarinFold/exp266/esm_documents/*.parquet",
        "s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/tokenized/mpnn-esm/2026.09.09.1",
        130_872_044,
        3338,
        138_755_354_859,
    ),
)

TRIALS = {
    trial.trial_id: trial
    for trial in (
        SizeTrial(
            trial_id="0_7b",
            label="0.7B",
            model=qwen3_contacts_config(hidden_dim=1536, num_layers=20, num_heads=24),
            nodes=8,
        ),
        SizeTrial(
            trial_id="1_5b",
            label="1.5B",
            model=qwen3_contacts_config(hidden_dim=2048, num_layers=24, num_heads=32),
            nodes=8,
        ),
        SizeTrial(
            trial_id="3b",
            label="3B",
            model=qwen3_contacts_config(hidden_dim=2560, num_layers=32, num_heads=40),
            nodes=8,
        ),
    )
}

VALIDATION_CACHE = "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/tokenized/contacts-v1-val/2026.07.25"
