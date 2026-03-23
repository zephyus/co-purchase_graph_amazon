#!/usr/bin/env bash
set -uo pipefail

PYTHON_BIN="/home/russell512/.venv/bin/python"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT_DIR"

OUT_A="results/q4_ultra_gpu0_bce"
OUT_B="results/q4_ultra_gpu1_bce"

mkdir -p "$OUT_A" "$OUT_B"

echo "[Ultra-Q4] Launching two parallel jobs to saturate both GPUs"

echo "[GPU0] BCE sweep"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" q4_advanced_link_prediction.py \
  --dataset-dir . \
  --output-dir "$OUT_A" \
  --device cuda \
  --seed 2026 \
  --epochs 170 \
  --patience 30 \
  --num-trials 12 \
  --train-pos-sample-ratio 0.60 \
  --loss-type bce \
  --bce-pos-weight 1.15 \
  --select-by f1 \
  --blend-heuristic \
  > "$OUT_A/train.log" 2>&1 &
PID_A=$!

echo "[GPU1] BCE sweep (different seed/weight)"
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" q4_advanced_link_prediction.py \
  --dataset-dir . \
  --output-dir "$OUT_B" \
  --device cuda \
  --seed 3407 \
  --epochs 170 \
  --patience 30 \
  --num-trials 12 \
    --train-pos-sample-ratio 0.70 \
    --loss-type bce \
    --bce-pos-weight 1.00 \
    --select-by f1 \
    --blend-heuristic \
  > "$OUT_B/train.log" 2>&1 &
PID_B=$!

echo "Waiting for jobs: PID_A=$PID_A PID_B=$PID_B"
STATUS_A=0
STATUS_B=0
wait "$PID_A" || STATUS_A=$?
wait "$PID_B" || STATUS_B=$?
echo "Job exit codes -> GPU0: $STATUS_A, GPU1: $STATUS_B"

echo "Both jobs completed. Comparing final metrics..."

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

candidates = [
    Path('results/q4_ultra_gpu0_bce/q4_metrics.json'),
    Path('results/q4_ultra_gpu1_bce/q4_metrics.json'),
]
rows = []
for p in candidates:
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    rows.append({
        'path': str(p),
        'auc': float(d['test']['auc']),
        'f1': float(d['test']['f1']),
        'precision': float(d['test']['precision']),
        'recall': float(d['test']['recall']),
        'pass_both': bool(d['baseline_targets']['pass_both']),
    })

if not rows:
    raise SystemExit('No q4_metrics.json found from ultra runs.')

rows_sorted = sorted(rows, key=lambda r: (r['pass_both'], r['f1'], r['auc']), reverse=True)
print('=== Q4 ULTRA RESULTS ===')
for r in rows_sorted:
    print(r)
print('BEST:', rows_sorted[0])
PY

echo "Ultra training finished. See logs in $OUT_A/train.log and $OUT_B/train.log"
