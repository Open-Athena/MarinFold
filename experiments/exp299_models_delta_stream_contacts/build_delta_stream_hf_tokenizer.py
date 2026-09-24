"""Write the contiguous HF tokenizer for delta-stream checkpoint export."""

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from convert_exp277_caches_to_delta_stream import (
    CONTACTS_BEGIN_TOKEN_ID,
    DOC_END_TOKEN_ID,
    DOC_START_TOKEN_ID,
    MAX_ABS_DELTA,
    STOP_TOKEN_ID,
    UNKNOWN_AA_TOKEN_ID,
    VOCAB_SIZE,
    delta_to_token,
)
from delta_stream_rollout import AA_ORDER


def vocabulary() -> dict[str, int]:
    """Return the exact token-ID vocabulary used by the V2 packed cache."""
    vocab = {"<stop>": STOP_TOKEN_ID}
    vocab.update({f"<aa:{aa}>": index + 1 for index, aa in enumerate(AA_ORDER)})
    vocab.update(
        {
            "<doc_start>": DOC_START_TOKEN_ID,
            "<contacts_begin>": CONTACTS_BEGIN_TOKEN_ID,
            "<doc_end>": DOC_END_TOKEN_ID,
            "<aa:UNK>": UNKNOWN_AA_TOKEN_ID,
        }
    )
    for index in range(25, 32):
        vocab[f"<reserved:{index}>"] = index
    for offset in range(-MAX_ABS_DELTA, MAX_ABS_DELTA + 1):
        if offset:
            vocab[f"<delta:{offset}>"] = delta_to_token(offset)
    if set(vocab.values()) != set(range(VOCAB_SIZE)):
        raise ValueError("delta tokenizer vocabulary is not contiguous")
    return vocab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = Tokenizer(WordLevel(vocabulary(), unk_token="<aa:UNK>"))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<aa:UNK>",
        pad_token="<stop>",
        bos_token="<doc_start>",
        eos_token="<doc_end>",
    )
    fast.save_pretrained(args.out)
    print(f"wrote {len(fast)} tokens to {args.out}")


if __name__ == "__main__":
    main()
