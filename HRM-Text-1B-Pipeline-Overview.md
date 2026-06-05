# HRM-Text-1B Post-Training: Overview & Design Decisions

**Goal:** Hill-climb MATH benchmark from 56.2% using SFT → DAPO on HRM-Text-1B  
**Timeline:** One week | **Hardware:** Hyak Klone `gpu-h200` (H200, preemptible ckpt)  
**Full implementation doc:** `HRM-Text-1B-Complete-Pipeline.md` (in Downloads)

---

## What Went Wrong Before (Prior Experiment)

Three compounding failures caused the RL stage to receive reward=0 for every single rollout:

1. **Wrong SFT dataset.** DAPO-Math-17K uses `"Answer: 42"` format (AIME-style), not `\boxed{42}`. After SFT the model never wrote `\boxed{}`, so Math-Verify found no answers to evaluate.

2. **Token cap too short.** 2048-token cap cut off MATH level 4-5 solutions mid-sentence, before `\boxed{}` appeared. Every truncated response = reward 0.

3. **Intermediate `\boxed{}` values in OpenR1.** DeepSeek-R1 self-corrects mid-reasoning, writing `\boxed{5}` then `\boxed{7}`. A Math-Verify bug concatenated all occurrences into a set → always wrong. Fix: extract only the **last** `\boxed{}`.

**All three are fixed in the current plan.**

---

## Why TRL Cannot Be Used (SFT)

HRM uses **PrefixLM attention**: instruction tokens attend bidirectionally (like BERT), response tokens attend causally (like GPT). This is implemented via a custom FlashAttention kernel that takes `prefix_lens`/`causal_lens`/`cu_seqlens` metadata.

TRL SFTTrainer is causal-only. It applies causal attention to all tokens including instructions — the wrong pattern from pretraining. Using it would corrupt every gradient update.

**Use instead:** HRM's own `pretrain.py --config-name cfg_sft`, or a custom training loop using the HuggingFace checkpoint with `token_type_ids` passed explicitly.

At inference/training via HF API: `token_type_ids=1` on instruction tokens → bidirectional attention. Omitting it → pure causal → "noticeably worse logits" (HRM model card).

---

## Why vLLM Cannot Be Used (DAPO Rollouts)

vLLM's PagedAttention assumes standard causal attention everywhere. It cannot apply bidirectional attention to instruction prefix tokens. The HRM model card explicitly lists vLLM support as "currently in progress."

Using vLLM would generate rollouts under the wrong attention mode, then train the model toward that corrupted distribution.

**Use instead:** HuggingFace `model.generate()` with `token_type_ids=torch.ones_like(input_ids)` on all prompt tokens.

---

## HRM-Text-1B Architecture

### Nested Recurrent Computation

HRM is not a standard transformer. Before producing each output token, it runs **2 H-cycles × 3 L-steps = 8 computation passes** through two reused weight matrices:

- **H-module (slow/strategic):** High-level context and direction. Updates once per H-cycle.
- **L-module (fast/execution):** Fine-grained computation. Runs 3 times per H-cycle.
- **State injection:** `z_L = z_L + z_H` — additive coupling between modules.
- **Weight-tying:** The same H-module and L-module weights are reused across all passes. This gives 8 effective compute passes with 1B parameters — "depth for free."

This architecture is why HRM achieves 56.2% MATH at only 1B params. A standard 1B transformer gets one pass; HRM gets eight.

### PrefixLM Attention

Every training example: `{condition, instruction, response}` JSONL.
- **Instruction tokens:** bidirectional attention (model sees the full problem before writing)
- **Response tokens:** causal attention (generated autoregressively)
- **Training control:** `prefix_lens`/`causal_lens`/`cu_seqlens` tensors
- **Inference control:** `token_type_ids` (1=bidirectional prefix, 0=causal response)

This is why TRL breaks: it can't pass the right metadata to HRM's custom kernel.

### Condition Flags

Special tokens prepended to the instruction that were in all 40B pretraining tokens. They condition the model's output style:

| Flag | Token | Meaning |
|------|-------|---------|
| `cot` | `<\|object_ref_end\|>` | Write step-by-step reasoning |
| `synth` | `<\|quad_end\|>` | Synthetically generated response |
| `direct` | `<\|object_ref_start\|>` | Short direct answer (no reasoning) |
| `noisy` | — | Web-crawled formatting (never use) |

**Always use `"cot,synth"`** for both SFT and DAPO. DAPO is designed for long-CoT reasoning. Switching to `direct` after `cot` SFT creates distribution mismatch.

### Latent Reasoning vs Explicit CoT

HRM reasons internally via recurrent H/L state cycles AND produces explicit step-by-step text in the response field. These are orthogonal — not in conflict. The `cot` flag teaches the model to write explicit reasoning steps in its output, which is exactly what DAPO needs to evaluate `\boxed{}` answers. The internal recurrent computation happens regardless.

### Pretraining Configuration

| Setting | Value | Why it matters |
|---------|-------|---------------|
| Context size | **4096 tokens** | Sets max_seq_length and max_new_tokens ceiling |
| Global batch | **172,032 tokens/step** | Token-count batching (not sample-count) |
| Attention | FlashAttention 3 | Hopper-only (H100/H200 required) |
| BPTT | 5 steps (warmed up from 2) | Full 5-step BPTT safe during fine-tuning |

---

## HRM-Text Repo Structure

The repo has **native SFT support** via `cfg_sft` config. Key files:

| File | Purpose |
|------|---------|
| `pretrain.py` | Main training loop — works for pretraining AND SFT |
| `scripts/prepare_sft_data.py` | Pre-tokenizes JSONL data for the native training stack |
| `config/cfg_sft.yaml` | SFT Hydra config (use with `--config-name cfg_sft`) |
| `dataset_new.py` | Data loader handling `prefix_lens`/`causal_lens`/`cu_seqlens` |
| `multipack_sampler.py` | Token-count bin-packing batching |
| `models/flash_attention_prefixlm_v2.py` | Custom PrefixLM FlashAttention kernel |
| `conversion/convert_to_hf.py` | Export FSDP2 checkpoint → HuggingFace format |

### Native SFT launch: `torchrun pretrain.py --config-name cfg_sft`

Key flag: `weights_only_resume_from_ema=true` — loads EMA weights from pretraining checkpoint with a fresh optimizer.

### Two Implementation Paths for LoRA

**Path A (recommended): HuggingFace model + PEFT LoRA + custom training loop**
- Load `sapientinc/HRM-Text-1B` via `AutoModelForCausalLM` (requires `transformers >= 5.9.0`)
- Apply PEFT `LoraConfig` + `get_peft_model()`
- Write custom loop that passes `token_type_ids` correctly
- LoRA only modifies weight matrices — PrefixLM correctness is entirely the model's own business via `token_type_ids`

**Path B: Fork `pretrain.py` with `cfg_sft`**
- Uses native token packing and `prefix_lens/causal_lens` system
- No built-in LoRA support
- Requires pre-tokenized data from `prepare_sft_data.py`
- FSDP2 overhead not needed for single GPU

---

## Data

### SFT Dataset (~27.5K examples, 100% `\boxed{}` guaranteed)

| Source | Size | Why chosen |
|--------|------|-----------|
| **OpenR1-Math-220K** (filtered) | 20K | DeepSeek-R1 generated full CoT + `\boxed{}`. Filter by `correctness_math_verify=True`. Extract **last** `\boxed{}` only. |
| **MATH train split** | 7.5K | Ground truth for the exact benchmark we're targeting. Always has `\boxed{}`. |

**Why not DAPO-Math-17K for SFT:** Uses `"Answer: $N"` format — no `\boxed{}`, no CoT. This was the cause of the prior experiment failure.

**Why not NuminaMath-CoT as primary:** Uses `■` (Unicode black square) as an answer delimiter in some examples. Math-Verify cannot parse `■` → silent reward failures.

**Why not general instruction data:** NVIDIA/CMU research found a measurable **-5% math harm** from naive mixing of general instruction data into math SFT. LoRA already protects against forgetting.

### Data Quality Pipeline (mandatory steps in order)

1. Download OpenR1-Math-220K
2. Filter: `correctness_math_verify=True` only
3. Extract **last** `\boxed{}` from each solution (not first, not all)
4. Verify answer against ground truth with Math-Verify (sympy)
5. Add MATH train split (7.5K, already clean)
6. **Assert 100% `\boxed{}` coverage before saving** — this is the critical check that would have caught the prior failure
7. Save as `{condition, instruction, response}` JSONL

### DAPO Prompt Dataset (~35K problems, no responses needed)

Sources (problem text only — responses generated online):
- MATH train Level 3-5 (where HRM sometimes succeeds, sometimes fails)
- OpenR1 filtered problems
- DAPO-Math-17K problems (problem text is high-quality; just ignore its answer format)

**Why Level 3-5 only:** Level 1-2 → model already solves reliably → all-correct batches → zero gradient. Level 5+ → model rarely solves → all-wrong batches → zero gradient. Level 3-4 → mixed correct/wrong → actual learning signal.

---

## Stage 1: SFT with LoRA

### Why SFT Before RL

HRM-Text-1B is a raw base model — never instruction-tuned or RLHF'd. It produces `\boxed{}` inconsistently. SFT locks in the format: "always write step-by-step reasoning ending with `\boxed{answer}`." SFT does not teach new math knowledge (the model already knows 56.2% of MATH).

### Why Exactly 1 Epoch

Over-SFT (2+ epochs) narrows the model's output distribution — mode collapse. This removes the diversity DAPO needs to explore different reasoning strategies. Multiple SFT epochs actively degrade downstream RL outcomes even when post-SFT benchmark scores look good (arXiv 2510.01624, NeurIPS 2025 Workshop, 100+ models, 1M+ GPU-hours).

### LoRA — Why and How

**Why LoRA:** Full fine-tuning of 1B params + 5-step BPTT requires ~4× inference memory. LoRA keeps base weights frozen and adds tiny rank-16 adapters (~5M params, ~0.5% of 1B).

**Why rank 16:** HRM already knows the math. SFT is teaching output formatting, which has low intrinsic dimensionality. Rank 16 = 16 independent directions of change — sufficient for format adaptation.

**Target modules (attention + MLP projections, in both H-module and L-module):**
- `q_proj, k_proj, v_proj, o_proj` — attention projections
- `gate_proj, up_proj, down_proj` — SwiGLU MLP projections

**Why NOT recurrent state-injection weights (`z_L + z_H` coupling):**
- Projection LoRA already captures equivalent expressiveness for the main recurrent parameters (proven in ICML 2025, arXiv 2410.09016)
- Weight-tying compounding: the same adapter applied N times per forward pass accumulates in hard-to-predict ways
- CUDA kernel constraints make attaching LoRA to bundled recurrent state matrices non-trivial

**Why NOT embeddings or LM head:** Modifying these disrupts the vocabulary mapping built during 40B-token pretraining.

### Key Training Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Epochs | 1 | Over-SFT degrades downstream RL |
| max_seq_length | 4096 | HRM pretraining context_size |
| Batching | Token-count packing | HRM uses 172K tokens/step |
| Learning rate | 2e-5 | Standard for LoRA SFT |
| Gradient clip | 1.0 | Recurrent BPTT amplifies gradients |
| BPTT steps | 5 (full) | Checkpoint is stable at 5 steps |
| token_type_ids | Required | 1=instruction, 0=response; omitting = wrong attention |

---

## Stage 2: DAPO (Main RL Experiment)

### The Core Problem DAPO Solves

The dominant failure mode in RL at 1B scale is **entropy collapse**: the model's output distribution narrows until all 8 responses to any prompt are nearly identical. GRPO's group-relative advantage (which needs variance within a group) collapses to zero. Training stalls.

### DAPO Mechanism 1: Clip-Higher (ε_high = 0.28)

Standard GRPO clips probability changes symmetrically at ±0.20. DAPO raises the upper bound to 0.28, giving "exploration tokens" (low-probability tokens that lead to novel reasoning paths) more room to increase probability. This delays entropy collapse.

**Important:** Clip-higher delays, does not prevent entropy collapse. Monitor entropy and intervene when needed.

### DAPO Mechanism 2: Dynamic Sampling

Discards zero-variance batches (all 8 responses identical reward → zero gradient → no learning) and resamples until there's a mixed correct/wrong batch. Evidence at 1B scale is mixed — implement but ablate if training slows.

### Why β = 0.0 (No KL Penalty)

Standard GRPO penalizes the policy for drifting from the pretrained distribution. DAPO removes this: long-CoT reasoning RL requires the policy to diverge significantly from pretraining to develop new strategies. KL penalty actively fights this beneficial divergence.

Confirmed by: DAPO paper (arXiv 2503.14476) + AceReason-Nemotron 1.1 + Magistral — all remove KL for math RL. ProRL retains KL but only for 16,000+ GPU-hour runs; not applicable here.

### Key DAPO Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| ε_high | 0.28 | Delays entropy collapse (confirmed 3-0) |
| ε_low | 0.20 | Same as vanilla GRPO |
| β_kl | 0.0 | No KL penalty for math RL (confirmed 3-0) |
| group_size | 8 | Rollouts per prompt |
| dynamic_sampling | True | Discard zero-variance batches (ablate if slow) |
| Reference resets | Every 250 steps | Prevents KL drift accumulation (ProRL) |
| max_new_tokens | 4096 | Hard limit from HRM context_size |
| Rollout generation | HF generate() | NOT vLLM — incompatible with PrefixLM |
| Condition flag | cot,synth | CoT rollouts — DAPO designed for long-CoT |

### Reward Function

- Extract **last** `\boxed{}` from response (not first — intermediate self-correction values)
- If no `\boxed{}` found → reward = 0.0 (includes token-cap truncation)
- If `\boxed{}` found: use Math-Verify (sympy) for symbolic equivalence check
  - Why sympy not string match: `1/2`, `0.5`, `\frac{1}{2}` are all correct — string match fails on 2/3 of equivalent forms at MATH level 4-5
- Correct + `\boxed{}`: reward = 1.1 (1.0 correctness + 0.1 format bonus)
- Wrong + `\boxed{}`: reward = 0.1 (format bonus only — can't be gamed without being correct)

### Training Health Monitoring (every 50 steps)

| Metric | Healthy | Intervention if... |
|--------|---------|-------------------|
| `policy_entropy` | Stable or slowly decreasing | Monotonically → 0: raise ε_high to 0.35, add entropy bonus |
| `reward_mean` | Trending upward | Flat 200+ steps: check entropy/prompt difficulty |
| `boxed_rate` | > 0.8 | < 0.5: check condition flag and token cap |
| `truncation_rate` | < 0.3 | > 0.3: filter prompts to Level 3-4 only |
| `kl_from_ref` | Near 0 | > 0.5: consider adding β=0.01 temporarily |

---

## RL Methods — Why Each Was Chosen or Rejected

### ✅ DAPO — Main Experiment
Asymmetric clipping (ε_high=0.28) + no KL penalty + long-CoT rollouts. Confirmed 3-0 evidence on all key parameters.

### ✅ REINFORCE++ — Day 7 Comparison
Replaces GRPO's local group normalization (can blow up when group std ≈ 0) with global batch normalization across the full training batch. Different failure profile from DAPO — useful comparison.

### ❌ RAFT — Dropped
Iterative filtered SFT (generate N responses, keep correct ones, fine-tune). DART-Math (NeurIPS 2024) showed 51.1% of Level-5 MATH problems get zero correct responses from rejection sampling. At 56.2% base, HRM already near RAFT's plateau. One epoch of DAPO gives more signal.

### ❌ Vanilla GRPO — Use DAPO Instead
DAPO is a free upgrade — two config numbers changed. No reason to run vanilla GRPO.

### ❌ PPO — Week 2+ Research
Requires 4 model copies (actor, critic, reference, reward model). Engineering a critic that understands HRM's recurrent state space (H-cycle and L-step level value estimation) is genuinely novel research — weeks, not days. Theoretically interesting for HRM specifically, not feasible in 1 week.

### ❌ DPO — Wrong Tool
DPO needs preference pairs (A preferred over B). Math has verifiable binary truth (correct/wrong). DAPO's binary reward signal is more direct. DPO can't be told "produce more correct answers" — only "prefer A over B," even if both are wrong.

### ❌ SDPO (arXiv 2601.20802) — Structurally Incompatible
Self-distillation policy optimization — uses rich textual feedback (error messages, execution traces) as self-teacher signal. Math-Verify returns a binary scalar. No textual feedback exists. Also: standalone SDPO underperforms vanilla GRPO at the 10-hour mark (71.1% vs 74.0%, confirmed 3-0 from SRPO paper). Not in TRL/veRL. Rejected.

### ❌ GTPO / GRPO-S — Invalidated
All specific performance claims refuted 0-3 in adversarial verification across 3 separate research passes. Do not use.

### ❌ TRL SFTTrainer — Architecturally Incompatible
Causal-only — corrupts PrefixLM bidirectional prefix attention. See Part 0.

### ❌ vLLM for Rollouts — Architecturally Incompatible
PagedAttention is causal-only. Listed "in progress" on HRM model card. See Part 0.

### ❌ General Data Mixing in SFT — Actively Harmful
NVIDIA/CMU research: naive mixed-quality SFT scaling = **-5% math benchmark harm** (not neutral — worse). LoRA already protects against forgetting general capabilities.

---

## Hardware & Infrastructure

### Why H100/H200 (Not A100, Not L40S)

HRM uses FlashAttention 3 — implemented with Hopper-specific tensor core operations that do not exist on Ampere (A100) or Ada Lovelace (L40S). Check the repo for FA2 fallback; if none exists, H100/H200 is required.

L40S also lacks NVLink — inter-GPU PCIe communication is ~10× slower than NVLink for any distributed training.

### Memory Footprint (Single H200, 141GB)

| Component | ~Memory |
|-----------|---------|
| Model weights (1B bf16) | 2 GB |
| LoRA adapters (rank 16) | 64 MB |
| Adam states (LoRA only) | 128 MB |
| BPTT activations (5 steps, 4096 ctx) | 15–25 GB |
| KV cache (DAPO generation) | 5–10 GB |
| **Total** | **~22–37 GB** |

H200 141GB has ~4× headroom. No FSDP2 needed — single GPU.

### FlashAttention 3 Installation

NOT available via `pip install flash-attn` (that gives FA2). Must compile from source:
- Clone flash-attention repo → navigate to `hopper/` subdirectory → `python setup.py install`
- Alternative: `pip install --pre flash-attn-4` (JIT-based, no compilation needed)
- Requires: H100 or H200 GPU, CUDA ≥ 12.3, ninja installed

### Hyak gpu-h200 Partition

- H200 GPUs on Hyak Klone via `-p gpu-h200`
- **Preemptible ckpt partition** — killed when priority users need the node
- ckpt GPU walltime limit: **8-9 hours**
- Preemption handling: `--requeue` + self-resubmit loop in sbatch script + MAX_RUNTIME budget (checkpoint 15 min before walltime)
- Storage: `/mmfs1/gscratch/intelligentsystems/$USER/` — Spectrum Scale GPFS, 100Gbps

---

## Key Numbers Quick Reference

| Parameter | Value |
|-----------|-------|
| Base MATH score | 56.2% |
| SFT dataset | ~27.5K examples |
| SFT epochs | 1 |
| Context / max_seq_length | 4096 tokens |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| DAPO ε_high | 0.28 |
| DAPO ε_low | 0.20 |
| DAPO β_kl | 0.0 |
| DAPO group_size | 8 |
| DAPO max_new_tokens | 4096 |
| Reference reset interval | 250 steps |
| Gradient clip norm | 1.0 |
| SFT learning rate | 2e-5 |
| DAPO condition flag | `cot,synth` |
| Transformers version required | ≥ 5.9.0 |

---

## One-Week Timeline

| Day | Activities |
|-----|-----------|
| **1 (Mon)** | Confirm `gpu-h200` access (`sinfo -p gpu-h200`), compile FA3, run data pipeline (filter OpenR1, assert 100% `\boxed{}`), inspect LoRA target param names, submit SFT |
| **2 (Tue)** | Monitor SFT, eval post-SFT MATH baseline (~200 problems), build DAPO prompt pool (~35K Level 3-5 problems), spot-check rollout quality |
| **3–5 (Wed–Fri)** | Submit DAPO, monitor entropy/boxed_rate/truncation_rate every 50 steps, handle preemptions via resubmit loop |
| **6 (Sat)** | Eval DAPO at step 500 and 1000 on full MATH, compare vs base and post-SFT |
| **7 (Sun)** | Submit REINFORCE++ from same SFT checkpoint, final eval: DAPO vs REINFORCE++ vs base |

---

## Files to Create (Not in Repo)

| File | What it does |
|------|-------------|
| `sft/train_hrm_sft.py` | Custom SFT: HF model + PEFT LoRA + training loop with `token_type_ids` |
| `rl/train_dapo.py` | DAPO: HF generate() rollouts + DAPO loss + reward function |
| `rl/train_reinforce_pp.py` | REINFORCE++ variant with global batch normalization |
| `data/prepare_data.py` | OpenR1 filtering + format conversion + 100% `\boxed{}` assertion |
| `eval/evaluate_math.py` | MATH benchmark evaluation helper |

## Files That Already Exist (Use These)

| File | What it does |
|------|-------------|
| `pretrain.py` | Native training loop (also runs SFT via `cfg_sft`) |
| `scripts/prepare_sft_data.py` | Tokenizes JSONL for native training stack |
| `config/cfg_sft.yaml` | SFT Hydra config |
| `models/flash_attention_prefixlm_v2.py` | PrefixLM custom kernel — do not modify |
| `conversion/convert_to_hf.py` | Export checkpoint to HuggingFace format |

---

## Confidence Notation
- **High (3-0):** Confirmed unanimously by adversarial verification across 3 independent agents
- **Medium (2-1):** Confirmed with one dissenter — treat as default, plan ablation
- **Invalidated (0-3):** Refuted — do not use
