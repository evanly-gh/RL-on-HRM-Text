#!/bin/bash
#SBATCH --job-name=hrm-sft
#SBATCH --partition=gpu-h200
#SBATCH --account=intelligentsystems-ckpt
#SBATCH --qos=ckpt-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/hrm-rl-2026/runs/%x-%j.out
#SBATCH --error=/mmfs1/gscratch/intelligentsystems/evanly/hrm-rl-2026/runs/%x-%j.err
set -euo pipefail

# ── New, clearly-named workspace (NOT the old gscratch run dirs) ──────────────
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/hrm-rl-2026
SFT_OUT=${SFT_OUT:-$PROJ/runs/sft-synthcot-v1}
REPO=${REPO:-/mmfs1/home/evanly/RL-on-HRM-Text}
ENV=${ENV:-/mmfs1/gscratch/intelligentsystems/evanly/conda_envs/rl}
DATA=${DATA:-$REPO/sft-data/sft_20k_synthcot.jsonl}
mkdir -p "$SFT_OUT/logs"

export HF_HOME=$PROJ/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$ENV/bin/python

echo "[sft] job=${SLURM_JOB_ID:-local} node=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1)"

# ── Wall-clock budget: exit 15 min before SLURM hard-kill to save a checkpoint ─
MARGIN=900; MAX_RUNTIME=0
END_RAW=$(scontrol show job "${SLURM_JOB_ID}" -o 2>/dev/null | grep -oE 'EndTime=[^ ]+' | cut -d= -f2 || true)
if [[ -n "${END_RAW:-}" && "${END_RAW}" != "Unknown" ]]; then
  END_EPOCH=$(date -d "${END_RAW}" +%s 2>/dev/null || echo 0); NOW=$(date +%s)
  [[ "${END_EPOCH}" -gt "${NOW}" ]] && MAX_RUNTIME=$(( END_EPOCH - NOW - MARGIN ))
  [[ "${MAX_RUNTIME}" -lt 60 ]] && MAX_RUNTIME=60
fi
echo "[sft] MAX_RUNTIME=${MAX_RUNTIME}s"

srun "$PY" "$REPO/sft/train_hrm_sft.py" \
  --data "$DATA" \
  --out "$SFT_OUT" \
  --epochs 1 --lr 2e-5 \
  --lora-r 16 --lora-alpha 16 --lora-dropout 0.05 \
  --max-len 4096 --max-tokens-per-batch 16384 --grad-accum 2 \
  --save-every 200 --log-every 10 \
  --max-runtime-seconds "${MAX_RUNTIME}" &
TRAIN_PID=$!
trap 'kill -USR1 "$TRAIN_PID" 2>/dev/null || true' USR1
wait "$TRAIN_PID"; RC=$?
echo "[sft] trainer exit=${RC}"

# ── Self-resubmit if not complete and a checkpoint exists ─────────────────────
N=${SFT_RESUBMIT_N:-0}; MAXN=${SFT_RESUBMIT_MAX:-12}
if [[ -f "$SFT_OUT/COMPLETE" ]]; then
  echo "[sft] COMPLETE — done."
elif [[ "$RC" -eq 0 && -e "$SFT_OUT/latest" && "$N" -lt "$MAXN" ]]; then
  echo "[sft] resubmitting chain $N -> $((N+1))"
  sbatch --export=ALL,SFT_RESUBMIT_N=$((N+1)),SFT_OUT="$SFT_OUT",REPO="$REPO",ENV="$ENV",DATA="$DATA" \
    "$REPO/sft/submit_sft.sbatch"
else
  echo "[sft] not resubmitting (RC=$RC N=$N)."
fi
exit "$RC"
