"""Exercise real Qwen training, checkpoint resume, and tokenizer export cheaply.

This test uses synthetic contacts and random tiny weights, so its results are
engineering validation only. It logs no W&B run and makes no protein-accuracy
claim. Run on CPU, one GPU, or two CPU DDP processes.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
from marinfold.document_structures.contacts_v1_multi import BEGIN, END, FINAL, MULTI
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from common import identity, rows, write_json, write_rows
from corpus import LossProfile, build_example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--generation", action="store_true")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    words = ["<pad>", "<unk>", "<contacts-v1>", MULTI, "<begin_sequence>", "<ALA>", BEGIN, FINAL, END,
             "<contact>", *[f"<p{i}>" for i in range(16)]]
    raw = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>")
    model_path = args.work / "initial"
    torch.manual_seed(281)
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=128, intermediate_size=256,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=64, max_position_embeddings=8192, attention_dropout=0.0,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.convert_tokens_to_ids(END))
    Qwen3ForCausalLM(config).save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    for split in ("train", "validation"):
        records = []
        for i in range(80):
            records.append(build_example(
                header=[MULTI, "<begin_sequence>", "<p0>", "<ALA>"],
                history=[BEGIN, "<contact>", "<p0>", "<p7>"] * (1 + i % 3),
                reference=["<contact>", "<p0>", "<p8>"] * (1 + i % 2),
                forced=i % 2 == 0, profile=LossProfile(), tokenizer=tokenizer, context=128))
        path = args.work / f"{split}.parquet"
        write_rows(str(path), records)
        write_json(str(args.work / f"{split}.json"), {"shards": [str(path)], "tokenizer_hash": identity(tokenizer.get_vocab())})
    env = dict(os.environ)
    if args.cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    command = ["uv", "run", "--no-project", sys.executable]
    if args.processes > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.processes}"]
    command += [str(Path(__file__).with_name("train.py")), "--model", str(model_path),
                "--train-manifest", str(args.work / "train.json"),
                "--validation-manifest", str(args.work / "validation.json"),
                "--output", str(args.work / "durable"), "--run-name", "exp281-smoke", "--steps", "2",
                "--global-batch", "4", "--microbatch", "1", "--warmup", "0", "--context", "128",
                "--save-every", "1", "--eval-every", "1", "--eval-documents", "2",
                "--work", str(args.work / "train-work"), "--no-wandb"]
    subprocess.run(command, check=True, env=env)
    prefix = args.work / "durable/checkpoints/exp281-smoke"
    before = Qwen3ForCausalLM.from_pretrained(prefix / "step-2").state_dict()
    subprocess.run([*command, "--resume", str(prefix / "step-1")], check=True, env=env)
    after = Qwen3ForCausalLM.from_pretrained(prefix / "step-2").state_dict()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    loaded = PreTrainedTokenizerFast.from_pretrained(prefix / "step-2")
    assert loaded.get_vocab() == tokenizer.get_vocab()
    write_json(str(args.work / "smoke_result.json"), {"resume_bitwise_equal": True, "tokenizer_equal": True,
               "processes": args.processes, "cpu": args.cpu})
    print("PASS: two training steps, checkpoint resume bitwise identical, tokenizer co-located", flush=True)
    if args.generation:
        if args.cpu:
            raise ValueError("vLLM smoke requires a GPU")
        targets = str(args.work / "targets.parquet")
        write_rows(targets, [{"target_id": "synthetic-smoke", "header": ["<contacts-v1>", "<begin_sequence>", "<p0>", "<ALA>"],
                             "reference": ["<contact>", "<p0>", "<p7>"], "positions": list(range(8)), "n_residues": 8}])
        target_manifest = str(args.work / "targets.json")
        write_json(target_manifest, {"shards": {"train": [targets]}})
        subprocess.run(["uv", "run", "--no-project", sys.executable, str(Path(__file__).with_name("generate.py")),
                        "--model", str(prefix / "step-2"), "--targets", target_manifest,
                        "--output", str(args.work / "generation"), "--context", "512", "--forced-fraction", "0",
                        "--candidates", "2", "--enforce-eager"], check=True, env=env)
        candidates = list(rows(str(args.work / "generation/shard-00000-part-00000.parquet")))
        assert len(candidates) == 2 and all("error" in c for c in candidates)
        print("PASS: vLLM batched generation, invalid-output accounting, and timing artifacts", flush=True)


if __name__ == "__main__":
    main()
