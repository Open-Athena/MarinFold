# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""exp163 Phase-2 MVP: LoRA fine-tune E8 on the refinement corpus, LOCAL, as a
fast go/no-go signal (does the refiner learn to USE candidates?). The real
1M-protein run goes through the TPU default_train path; this only validates the
method cheaply.

LM loss is masked to the <begin_statements> answer section only: the model is
trained to emit the TRUE contacts given (sequence + K noisy candidate blocks),
never to reproduce the sequence or the noisy candidates.

    uv run --with peft python mvp_finetune.py --corpus corpus_train.jsonl \
        --model /home/bizon/exp89_export/hf_step35679 --out mvp_refiner --smoke
"""
from __future__ import annotations
import argparse, json
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)
from peft import LoraConfig, get_peft_model

BEGIN = "<begin_statements>"

class RefineDS(Dataset):
    """Tokenize each doc; labels = -100 except the answer contacts after BEGIN."""
    def __init__(self, path, tok, max_len, max_docs=None):
        self.ex = []
        begin_id = tok.convert_tokens_to_ids(BEGIN)
        eos = tok.eos_token_id
        dropped = 0
        for line in open(path):
            text = json.loads(line)["text"]
            ids = tok(text, add_special_tokens=False).input_ids + [eos]
            if len(ids) > max_len:
                dropped += 1
                continue
            if begin_id not in ids:
                continue
            b = ids.index(begin_id)
            labels = [-100] * (b + 1) + ids[b + 1:]          # loss only after BEGIN
            self.ex.append({"input_ids": ids, "labels": labels})
            if max_docs and len(self.ex) >= max_docs:
                break
        self.dropped = dropped
        print(f"dataset: {len(self.ex)} docs (dropped {dropped} > max_len={max_len})", flush=True)

    def __len__(self): return len(self.ex)
    def __getitem__(self, i): return self.ex[i]

def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    ii, ll, am = [], [], []
    for b in batch:
        n = m - len(b["input_ids"])
        ii.append(b["input_ids"] + [pad_id] * n)
        ll.append(b["labels"] + [-100] * n)
        am.append([1] * len(b["input_ids"]) + [0] * n)
    return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
            "attention_mask": torch.tensor(am)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", default="/home/bizon/exp89_export/hf_step35679")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate end-to-end")
    a = ap.parse_args()
    set_seed(0)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = RefineDS(a.corpus, tok, a.max_len, max_docs=64 if a.smoke else None)

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out + "_ckpt", per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs,
        learning_rate=a.lr, bf16=True, gradient_checkpointing=True,
        warmup_ratio=0.03, lr_scheduler_type="cosine", logging_steps=5,
        save_strategy="no", report_to=[], max_steps=4 if a.smoke else -1,
        dataloader_num_workers=2, optim="adamw_torch",
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    trainer.train()

    print("merging LoRA + saving ...", flush=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"saved merged refiner -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
