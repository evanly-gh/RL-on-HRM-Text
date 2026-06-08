#!/bin/bash
#SBATCH --job-name=hrm-dapo
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

PROJ=/mmfs1/gscratch/intelligentsystems/evanly/hrm-rl-2026
OUT=${OUT:-$PROJ/runs/dapo-v3}
REPO=${REPO:-/mmfs1/home/evanly/RL-on-HRM-Text}
ENV=${ENV:-/mmfs1/gscratch/intelligentsystems/evanly/conda_envs/rl}
SFT_ADAPTER=${SFT_ADAPTER:-$PROJ/runs/sft-synthcot-v1/final_adapter}
PROMPTS=${PROMPTS:-$REPO/rl-data/hendrycks_l35_dapo.jsonl}
mkdir -p "$OUT/logs"

export HF_HOME=$PROJ/hf_cache
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$ENV/bin/python

echo "[dapo] job=${SLURM_JOB_ID:-local} node=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1)"

MARGIN=900; MAX_RUNTIME=0
END_RAW=$(scontrol show job "${SLURM_JOB_ID}" -o 2>/dev/null | grep -oE 'EndTime=[^ ]+' | cut -d= -f2 || true)
if [[ -n "${END_RAW:-}" && "${END_RAW}" != "Unknown" ]]; then
  END_EPOCH=$(date -d "${END_RAW}" +%s 2>/dev/null || echo 0); NOW=$(date +%s)
  [[ "${END_EPOCH}" -gt "${NOW}" ]] && MAX_RUNTIME=$(( END_EPOCH - NOW - MARGIN ))
  [[ "${MAX_RUNTIME}" -lt 60 ]] && MAX_RUNTIME=60
fi
echo "[dapo] MAX_RUNTIME=${MAX_RUNTIME}s"

srun "$PY" "$REPO/rl/train_dapo.py" \
  --sft-adapter "$SFT_ADAPTER" \
  --prompts "$PROMPTS" \
  --out "$OUT" \
  --total-steps 80 \
  --prompts-per-step 16 --group 8 \
  --min-kept-groups 6 --max-resample 1 \
  --max-new-tokens 3072 --temperature 1.0 \
  --gen-micro 32 --train-micro 4 --ppo-epochs 2 \
  --lr 5e-6 --eps-low 0.20 --eps-high 0.28 \
  --save-every 5 \
  --max-runtime-seconds "${MAX_RUNTIME}" &
TRAIN_PID=$!
trap 'kill -USR1 "$TRAIN_PID" 2>/dev/null || true' USR1
wait "$TRAIN_PID"; RC=$?
echo "[dapo] trainer exit=${RC}"

N=${DAPO_RESUBMIT_N:-0}; MAXN=${DAPO_RESUBMIT_MAX:-60}
# Resubmit on ANY exit (incl. preemption RC=137/143) as long as a checkpoint
# exists and we're not COMPLETE — ckpt partition preempts often, and preemption
# is a non-zero exit that must NOT break the chain.
if [[ -f "$OUT/COMPLETE" ]]; then
  echo "[dapo] COMPLETE."
elif [[ -e "$OUT/latest" && "$N" -lt "$MAXN" ]]; then
  echo "[dapo] resubmit chain $N -> $((N+1)) (RC=$RC)"
  sbatch --export=ALL,DAPO_RESUBMIT_N=$((N+1)),OUT="$OUT",REPO="$REPO",ENV="$ENV",SFT_ADAPTER="$SFT_ADAPTER",PROMPTS="$PROMPTS" \
    "$REPO/rl/submit_dapo.sbatch"
else
  echo "[dapo] not resubmitting (RC=$RC N=$N latest=$([ -e "$OUT/latest" ] && echo y || echo n))."
fi
exit "$RC"
