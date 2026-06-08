#!/usr/bin/env python
"""HRM-Text-1B SFT via HuggingFace model + PEFT LoRA (Path A).

Why this and not TRL SFTTrainer: HRM is a PrefixLM. The instruction (prefix)
tokens must attend bidirectionally and the response tokens causally. That is
controlled by `token_type_ids` (==1 prefix / ==0 response), which the HF
`hrm_text` model honors only when `config.prefix_lm=True` (it is). TRL applies
causal attention everywhere and cannot pass token_type_ids -> wrong attention
distribution. So we run our own loop.

Sequence layout (matches scripts/prepare_sft_data.py / pretraining):
    [<|im_start|>] [cond tokens...] [instruction ids] [<|im_end|>]   <- prefix, type=1, label=-100
    [response ids] [<|box_end|>]                                     <- response, type=0, supervised

Loss is computed only on response tokens (task-completion NLL).

Checkpointing is preemption-safe (Hyak ckpt partition): adapter + optimizer +
scheduler + step + RNG are saved every --save-every steps and on time-budget
exit (clean exit code 0 so the sbatch self-resubmit loop continues). A COMPLETE
sentinel is written when the full run finishes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

# ---- special tokens (HRM-Text-1B tokenizer) --------------------------------
BOQ = 6          # <|im_start|>   begin instruction
EOQ = 7          # <|im_end|>     end instruction
EOA = 11         # <|box_end|>    end answer (also eos)
PAD = 5          # <|endoftext|>
COND_TOKENS = {  # condition label -> token id
    "direct": 8,    # <|object_ref_start|>
    "cot": 9,       # <|object_ref_end|>
    "noisy": 12,    # <|quad_start|>
    "synth": 13,    # <|quad_end|>
}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def log(msg: str):
    print(f"[sft {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_examples(jsonl_path: str, tokenizer, max_len: int):
    """Tokenize every row into (input_ids, token_type_ids, labels). Returns list."""
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    examples = []
    dropped = 0
    for r in rows:
        cond_ids = [COND_TOKENS[c.strip()] for c in r.get("condition", "direct").split(",")]
        inst_ids = tokenizer(r["instruction"], add_special_tokens=False)["input_ids"]
        resp_ids = tokenizer(r["response"], add_special_tokens=False)["input_ids"]
        if not resp_ids:
            dropped += 1
            continue
        prefix = [BOQ] + cond_ids + inst_ids + [EOQ]
        response = resp_ids + [EOA]
        input_ids = prefix + response
        if len(input_ids) > max_len:
            dropped += 1
            continue
        ttids = [1] * len(prefix) + [0] * len(response)
        labels = [-100] * len(prefix) + response
        examples.append((input_ids, ttids, labels))
    log(f"built {len(examples)} examples ({dropped} dropped: empty/over-length)")
    return examples


def make_batches(examples, max_tokens_per_batch: int):
    """Length-grouped batching: sort by length, greedily pack so that
    (batch_size * longest_in_batch) <= max_tokens_per_batch. Returns list of
    index lists (into `examples`)."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    batches, cur = [], []
    for i in order:
        L = len(examples[i][0])
        if cur and (len(cur) + 1) * max(L, len(examples[cur[0]][0])) > max_tokens_per_batch:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def collate(idxs, examples, device):
    seqs = [examples[i] for i in idxs]
    maxlen = max(len(s[0]) for s in seqs)
    B = len(seqs)
    input_ids = torch.full((B, maxlen), PAD, dtype=torch.long)
    ttids = torch.zeros((B, maxlen), dtype=torch.long)
    labels = torch.full((B, maxlen), -100, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    for b, (ii, tt, ll) in enumerate(seqs):
        n = len(ii)
        input_ids[b, :n] = torch.tensor(ii)
        ttids[b, :n] = torch.tensor(tt)
        labels[b, :n] = torch.tensor(ll)
        attn[b, :n] = 1
    return (input_ids.to(device), ttids.to(device), labels.to(device), attn.to(device))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_ckpt(out_dir: Path, model, optim, sched, step, epoch, batch_idx, rng_state):
    ck = out_dir / f"checkpoint-{step}"
    ck.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ck)  # LoRA adapter only
    torch.save(optim.state_dict(), ck / "optimizer.pt")
    torch.save(sched.state_dict(), ck / "scheduler.pt")
    torch.save(rng_state, ck / "rng.pt")
    with open(ck / "trainer_state.json", "w") as f:
        json.dump({"step": step, "epoch": epoch, "batch_idx": batch_idx}, f)
    # update 'latest' pointer
    latest = out_dir / "latest"
    tmp = out_dir / "latest.tmp"
    tmp.write_text(ck.name)
    tmp.replace(latest)
    # keep only the 2 most recent checkpoints
    cks = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    for old in cks[:-2]:
        for p in old.iterdir():
            p.unlink()
        old.rmdir()
    log(f"saved {ck}")


def find_latest(out_dir: Path):
    latest = out_dir / "latest"
    if latest.exists():
        ck = out_dir / latest.read_text().strip()
        if ck.exists():
            return ck
    cks = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return cks[-1] if cks else None


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="sapientinc/HRM-Text-1B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-tokens-per-batch", type=int, default=16384)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-runtime-seconds", type=int, default=0,
                    help="If >0, save & exit(0) once exceeded (preemption budget).")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="If >0, stop after this many optimizer steps (smoke test).")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "COMPLETE").exists():
        log("COMPLETE sentinel present — nothing to do."); return

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    log("loading base model (bf16)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True)
    base.config.use_cache = False
    base.gradient_checkpointing_enable()

    resume_ck = find_latest(out_dir)
    if resume_ck is not None:
        log(f"resuming LoRA from {resume_ck}")
        model = PeftModel.from_pretrained(base, resume_ck, is_trainable=True)
    else:
        lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                          lora_dropout=args.lora_dropout, bias="none",
                          task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
        model = get_peft_model(base, lora)
    model.to(device)
    model.print_trainable_parameters()

    examples = build_examples(args.data, tokenizer, args.max_len)
    batches = make_batches(examples, args.max_tokens_per_batch)
    steps_per_epoch = math.ceil(len(batches) / args.grad_accum)
    total_opt_steps = steps_per_epoch * args.epochs
    log(f"{len(batches)} micro-batches/epoch | {steps_per_epoch} opt-steps/epoch | "
        f"{total_opt_steps} total opt-steps")

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(
        optim, int(args.warmup_ratio * total_opt_steps), total_opt_steps)

    start_epoch, start_batch, gstep = 0, 0, 0
    if resume_ck is not None:
        # weights_only=False: these trusted checkpoints hold non-tensor (numpy RNG) state
        optim.load_state_dict(torch.load(resume_ck / "optimizer.pt", map_location=device, weights_only=False))
        sched.load_state_dict(torch.load(resume_ck / "scheduler.pt", map_location="cpu", weights_only=False))
        st = json.loads((resume_ck / "trainer_state.json").read_text())
        gstep, start_epoch, start_batch = st["step"], st["epoch"], st["batch_idx"]
        rng = torch.load(resume_ck / "rng.pt", weights_only=False)
        random.setstate(rng["py"]); np.random.set_state(rng["np"]); torch.set_rng_state(rng["torch"])
        log(f"resumed at epoch={start_epoch} batch_idx={start_batch} gstep={gstep}")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        rng_epoch = random.Random(args.seed + epoch)
        batch_order = list(range(len(batches)))
        rng_epoch.shuffle(batch_order)
        running = 0.0
        optim.zero_grad(set_to_none=True)
        for bpos in range(len(batch_order)):
            if epoch == start_epoch and bpos < start_batch:
                continue
            idxs = batches[batch_order[bpos]]
            input_ids, ttids, labels, attn = collate(idxs, examples, device)
            out = model(input_ids=input_ids, token_type_ids=ttids,
                        attention_mask=attn, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item()

            if (bpos + 1) % args.grad_accum == 0 or bpos == len(batch_order) - 1:
                gnorm = clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                        args.grad_clip)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % args.log_every == 0:
                    avg = running / args.log_every / max(1, args.grad_accum)
                    log(f"ep{epoch} step{gstep}/{total_opt_steps} loss={avg:.4f} "
                        f"lr={sched.get_last_lr()[0]:.2e} gnorm={gnorm:.2f} "
                        f"elapsed={int(time.time()-t0)}s")
                    running = 0.0
                if gstep % args.save_every == 0:
                    rng = {"py": random.getstate(), "np": np.random.get_state(),
                           "torch": torch.get_rng_state()}
                    save_ckpt(out_dir, model, optim, sched, gstep, epoch, bpos + 1, rng)
                if args.max_steps and gstep >= args.max_steps:
                    log(f"--max-steps {args.max_steps} reached — smoke test done, exiting.")
                    return

            if args.max_runtime_seconds and (time.time() - t0) > args.max_runtime_seconds:
                log("time budget reached — saving and exiting(0) for resubmit.")
                rng = {"py": random.getstate(), "np": np.random.get_state(),
                       "torch": torch.get_rng_state()}
                save_ckpt(out_dir, model, optim, sched, gstep, epoch, bpos + 1, rng)
                return
        start_batch = 0  # next epoch starts fresh

    # done
    final = out_dir / "final_adapter"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    (out_dir / "COMPLETE").write_text(f"done at step {gstep}\n")
    log(f"TRAINING COMPLETE — adapter at {final}")


if __name__ == "__main__":
    main()
