#!/usr/bin/env python
"""DAPO (decoupled-clip GRPO) for HRM-Text-1B, Path A (HF generate rollouts).

Pipeline per step:
  1. sample N unique prompts; generate G rollouts each with token_type_ids=1 on
     the prompt (PrefixLM) — HF generate, NOT vLLM.
  2. reward each rollout via Math-Verify (last \\boxed{} vs gold). r in {0,1}.
  3. dynamic sampling: drop zero-variance groups (all right / all wrong).
  4. group-relative advantage A_i = (r_i - mean_g)/(std_g+eps), broadcast to the
     rollout's response tokens.
  5. DAPO surrogate with asymmetric clip (eps_low=0.20, eps_high=0.28), beta_kl=0,
     token-level mean over all response tokens. K ppo-epochs.

SFT init: the SFT LoRA is merged into the frozen base; a fresh LoRA trains under
DAPO. beta=0 => no reference model needed (kl logged optionally via base if asked,
but skipped here for memory).

Preemption-safe checkpointing identical in spirit to the SFT script.
"""
from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from reward import extract_last_boxed, is_correct

BOQ, EOQ, EOA, PAD = 6, 7, 11, 5
COND = {"direct": 8, "cot": 9, "noisy": 12, "synth": 13}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def log(m): print(f"[dapo {time.strftime('%H:%M:%S')}] {m}", flush=True)


def prompt_ids(tok, condition, instruction):
    cond = [COND[c.strip()] for c in condition.split(",")]
    return [BOQ] + cond + tok(instruction, add_special_tokens=False)["input_ids"] + [EOQ]


# ---- logprob/entropy over response tokens for a right-padded batch ----------
def seq_logprobs(model, input_ids, ttids, attn, resp_mask, want_entropy=False):
    """Return (logp_taken [B,T-1], entropy [B,T-1] or None) aligned so that index
    t corresponds to predicting input_ids[:,t+1]. resp_mask marks response targets."""
    out = model(input_ids=input_ids, token_type_ids=ttids, attention_mask=attn)
    logits = out.logits[:, :-1, :].float()          # predict token t+1
    targets = input_ids[:, 1:]
    logp_all = F.log_softmax(logits, dim=-1)
    logp_taken = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ent = None
    if want_entropy:
        ent = -(logp_all.exp() * logp_all).sum(-1)
    return logp_taken, ent


def collate_seqs(seqs, device):
    """seqs: list of (full_ids list, prompt_len). Right-pad. Returns tensors and
    resp_mask over targets (index t marks input_ids[t+1] is a response token)."""
    T = max(len(s[0]) for s in seqs)
    B = len(seqs)
    input_ids = torch.full((B, T), PAD, dtype=torch.long)
    ttids = torch.zeros((B, T), dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    resp_mask = torch.zeros((B, T - 1), dtype=torch.float)
    for b, (ids, plen) in enumerate(seqs):
        n = len(ids)
        input_ids[b, :n] = torch.tensor(ids)
        ttids[b, :plen] = 1
        attn[b, :n] = 1
        # response targets are positions plen..n-1 in input_ids -> target index plen-1..n-2
        resp_mask[b, plen - 1:n - 1] = 1.0
    return (input_ids.to(device), ttids.to(device), attn.to(device), resp_mask.to(device))


def save_ckpt(out_dir, model, optim, sched, step, rng):
    ck = out_dir / f"checkpoint-{step}"; ck.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ck)
    torch.save(optim.state_dict(), ck / "optimizer.pt")
    torch.save(sched.state_dict(), ck / "scheduler.pt")
    torch.save(rng, ck / "rng.pt")
    (ck / "trainer_state.json").write_text(json.dumps({"step": step}))
    tmp = out_dir / "latest.tmp"; tmp.write_text(ck.name); tmp.replace(out_dir / "latest")
    cks = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    for old in cks[:-2]:
        for p in old.iterdir(): p.unlink()
        old.rmdir()
    log(f"saved {ck}")


def find_latest(out_dir):
    lp = out_dir / "latest"
    if lp.exists():
        ck = out_dir / lp.read_text().strip()
        if ck.exists(): return ck
    cks = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return cks[-1] if cks else None


@torch.no_grad()
def generate_group(model, tok, batch_prompt_ids, group, max_new, temp, top_p, micro):
    """Return list of (full_ids, prompt_len) for each (prompt, rollout)."""
    items = []
    for pid in batch_prompt_ids:
        for _ in range(group):
            items.append(pid)
    results = []
    for s in range(0, len(items), micro):
        chunk = items[s:s + micro]
        T = max(len(p) for p in chunk); B = len(chunk)
        inp = torch.full((B, T), PAD, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        for b, p in enumerate(chunk):
            inp[b, T - len(p):] = torch.tensor(p); attn[b, T - len(p):] = 1
        inp = inp.cuda(); attn = attn.cuda()
        out = model.generate(input_ids=inp, attention_mask=attn, token_type_ids=attn.clone(),
                             max_new_tokens=max_new, do_sample=True, temperature=temp,
                             top_p=top_p, pad_token_id=PAD, eos_token_id=EOA)
        gen = out[:, T:]
        for b, p in enumerate(chunk):
            g = gen[b].tolist()
            if EOA in g:
                g = g[:g.index(EOA) + 1]
            else:
                while g and g[-1] == PAD: g.pop()
            results.append((list(p) + g, len(p)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sapientinc/HRM-Text-1B")
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-steps", type=int, default=1000)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--gen-micro", type=int, default=64)
    ap.add_argument("--train-micro", type=int, default=8, help="seqs per fwd/bwd")
    ap.add_argument("--ppo-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--eps-low", type=float, default=0.20)
    ap.add_argument("--eps-high", type=float, default=0.28)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--min-kept-groups", type=int, default=8,
                    help="collect at least this many non-degenerate groups per step")
    ap.add_argument("--max-resample", type=int, default=2,
                    help="max EXTRA prompt-batch draws per step to reach min-kept-groups")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-runtime-seconds", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "COMPLETE").exists(): log("already COMPLETE"); return
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    log("loading base + merging SFT adapter...")
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                trust_remote_code=True)
    base = PeftModel.from_pretrained(base, args.sft_adapter).merge_and_unload()
    base.config.use_cache = True

    resume = find_latest(out_dir)
    if resume is not None:
        log(f"resuming DAPO LoRA from {resume}")
        model = PeftModel.from_pretrained(base, resume, is_trainable=True)
    else:
        lc = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
        model = get_peft_model(base, lc)
    model.cuda()
    model.print_trainable_parameters()

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = get_constant_schedule_with_warmup(optim, 10)

    gstep = 0
    if resume is not None:
        # weights_only=False: optimizer/scheduler/rng contain non-tensor (numpy RNG)
        # state; torch>=2.6 defaults weights_only=True which rejects them. These are
        # our own trusted checkpoints.
        optim.load_state_dict(torch.load(resume / "optimizer.pt", map_location="cuda", weights_only=False))
        sched.load_state_dict(torch.load(resume / "scheduler.pt", map_location="cpu", weights_only=False))
        gstep = json.loads((resume / "trainer_state.json").read_text())["step"]
        rng = torch.load(resume / "rng.pt", weights_only=False)
        random.setstate(rng["py"]); np.random.set_state(rng["np"]); torch.set_rng_state(rng["torch"])
        log(f"resumed at step {gstep}")

    prompts = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    log(f"{len(prompts)} prompts in pool")
    metrics_log = open(out_dir / "metrics.jsonl", "a")

    while gstep < args.total_steps:
        # ---- collect a batch of non-degenerate groups ----
        kept = []  # (full_ids, prompt_len, advantage)
        raw_rewards, boxed_n, trunc_n, total_roll = [], 0, 0, 0
        draws = 0; kept_groups = 0
        model.eval()
        # Draw prompt batches until we have enough NON-DEGENERATE groups (dynamic
        # sampling keeps ~44% of groups), capped by max_resample extra draws.
        while kept_groups < args.min_kept_groups and draws <= args.max_resample:
            draws += 1
            batch = random.sample(prompts, args.prompts_per_step)
            pids = [prompt_ids(tok, b["condition"], b["instruction"]) for b in batch]
            rolls = generate_group(model, tok, pids, args.group, args.max_new_tokens,
                                   args.temperature, args.top_p, args.gen_micro)
            # rolls grouped consecutively by prompt
            for gi, b in enumerate(batch):
                grp = rolls[gi * args.group:(gi + 1) * args.group]
                rewards = []
                for full_ids, plen in grp:
                    gen_ids = full_ids[plen:]
                    text = tok.decode(gen_ids, skip_special_tokens=True)
                    lb = extract_last_boxed(text)
                    boxed_n += int(lb is not None)
                    if lb is None and (EOA not in gen_ids): trunc_n += 1
                    r = 1.0 if is_correct(text, b["answer"]) else 0.0
                    rewards.append(r); raw_rewards.append(r); total_roll += 1
                rewards = np.array(rewards)
                if rewards.std() < 1e-6:
                    continue  # dynamic sampling: drop zero-variance group
                adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
                kept_groups += 1
                for (full_ids, plen), a in zip(grp, adv):
                    kept.append((full_ids, plen, float(a)))

        if not kept:
            log(f"step {gstep}: no non-degenerate groups in {draws} draws "
                f"(boxed_rate={boxed_n/max(1,total_roll):.2f} "
                f"reward_mean={np.mean(raw_rewards) if raw_rewards else 0:.3f}); resampling.")
            continue

        # ---- old logprobs (current policy, detached) ----
        model.eval()
        old_lp = [None] * len(kept)
        advs = torch.tensor([k[2] for k in kept], dtype=torch.float)
        with torch.no_grad():
            for s in range(0, len(kept), args.train_micro):
                seqs = [(k[0], k[1]) for k in kept[s:s + args.train_micro]]
                ii, tt, am, rm = collate_seqs(seqs, "cuda")
                lp, _ = seq_logprobs(model, ii, tt, am, rm)
                for j, (ids, _plen) in enumerate(seqs):
                    # store trimmed to this sequence's true target length (n-1),
                    # independent of batch padding, so PPO re-batching aligns.
                    old_lp[s + j] = lp[j, :len(ids) - 1].detach().cpu()

        # ---- PPO updates ----
        model.train()
        last = {}
        for _ep in range(args.ppo_epochs):
            order = list(range(len(kept))); random.shuffle(order)
            for s in range(0, len(order), args.train_micro):
                idx = order[s:s + args.train_micro]
                seqs = [(kept[i][0], kept[i][1]) for i in idx]
                ii, tt, am, rm = collate_seqs(seqs, "cuda")
                lp_new, ent = seq_logprobs(model, ii, tt, am, rm, want_entropy=True)
                ent_sum = 0.0
                tok_count = rm.sum().clamp_min(1.0)
                # build per-row clipped surrogate (DAPO asymmetric clip).
                # Slice each row to its true target length (n-1) so old/new/mask align.
                loss_terms = []
                for r, i in enumerate(idx):
                    n = len(kept[i][0]) - 1
                    old = old_lp[i].to("cuda")          # [n]
                    new = lp_new[r, :n]                  # [n]
                    m = rm[r, :n]                        # [n]
                    a = advs[i].to("cuda")
                    ratio = torch.exp(new - old)
                    unclipped = ratio * a
                    clipped = torch.clamp(ratio, 1 - args.eps_low, 1 + args.eps_high) * a
                    surr = torch.minimum(unclipped, clipped) * m
                    loss_terms.append(surr.sum())
                    ent_sum = ent_sum + (ent[r, :n] * m).sum()
                loss = -(torch.stack(loss_terms).sum()) / tok_count
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
                last = {"loss": float(loss.detach()), "gnorm": float(gnorm),
                        "entropy": float((ent_sum / tok_count).detach())}

        gstep += 1
        m = {"step": gstep, "reward_mean": float(np.mean(raw_rewards)),
             "reward_std": float(np.std(raw_rewards)),
             "boxed_rate": boxed_n / max(1, total_roll),
             "trunc_rate": trunc_n / max(1, total_roll),
             "kept_rollouts": len(kept), "kept_groups": kept_groups, "draws": draws,
             "mean_adv_abs": float(advs.abs().mean()), **last,
             "elapsed": int(time.time() - t0)}
        log(f"step {gstep}/{args.total_steps} reward={m['reward_mean']:.3f} "
            f"boxed={m['boxed_rate']:.2f} loss={last.get('loss',0):.4f} "
            f"ent={last.get('entropy',0):.3f} kept={len(kept)} draws={draws}")
        metrics_log.write(json.dumps(m) + "\n"); metrics_log.flush()

        if gstep % args.save_every == 0:
            rng = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
            save_ckpt(out_dir, model, optim, sched, gstep, rng)
        if args.max_runtime_seconds and (time.time() - t0) > args.max_runtime_seconds:
            log("time budget reached — checkpoint & exit(0).")
            rng = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
            save_ckpt(out_dir, model, optim, sched, gstep, rng)
            return

    model.save_pretrained(out_dir / "final_adapter")
    tok.save_pretrained(out_dir / "final_adapter")
    (out_dir / "COMPLETE").write_text(f"done at step {gstep}\n")
    log("DAPO COMPLETE")


if __name__ == "__main__":
    main()
