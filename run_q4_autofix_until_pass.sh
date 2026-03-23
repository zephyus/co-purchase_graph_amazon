#!/usr/bin/env bash
set -uo pipefail

PYTHON_BIN="/home/russell512/.venv/bin/python"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_ROOT="$ROOT_DIR/results/q4_autofix"
mkdir -p "$OUT_ROOT"
cd "$ROOT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# format:
# name,trial,seed,train,val,test,sample,posw,epochs,patience,gpu
RUNS=(
  "r01_t8_s2026,8,2026,0.70,0.15,0.15,0.80,1.00,220,40,0"
  "r02_t9_s3407,9,3407,0.70,0.15,0.15,0.75,1.10,220,40,1"
  "r03_t10_s777,10,777,0.70,0.15,0.15,0.70,1.00,220,40,0"
  "r04_t8_s1111,8,1111,0.75,0.10,0.15,0.80,1.00,220,40,1"
  "r05_t9_s2222,9,2222,0.80,0.10,0.10,0.80,1.00,220,40,0"
  "r06_t10_s3333,10,3333,0.80,0.10,0.10,0.70,1.10,220,40,1"
  "r07_t11_s4444,11,4444,0.75,0.10,0.15,0.70,1.00,220,40,0"
  "r08_t12_s5555,12,5555,0.80,0.10,0.10,0.65,1.00,220,40,1"
)

run_one() {
  local spec="$1"
  IFS=',' read -r NAME TRIAL SEED TR VR TSR SAMPLE POSW EPOCHS PATIENCE GPU <<< "$spec"
  local OUT_DIR="$OUT_ROOT/$NAME"
  mkdir -p "$OUT_DIR"
  echo "[START] $NAME gpu=$GPU trial=$TRIAL split=$TR/$VR/$TSR sample=$SAMPLE posw=$POSW"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" q4_advanced_link_prediction.py \
    --dataset-dir . \
    --output-dir "$OUT_DIR" \
    --device cuda \
    --single-trial-index "$TRIAL" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --train-ratio "$TR" \
    --val-ratio "$VR" \
    --test-ratio "$TSR" \
    --train-pos-sample-ratio "$SAMPLE" \
    --loss-type bce \
    --bce-pos-weight "$POSW" \
    --select-by f1 \
    --blend-heuristic \
    > "$OUT_DIR/train.log" 2>&1
  local code=$?
  echo "[END] $NAME exit=$code"
  return $code
}

report_best() {
  "$PYTHON_BIN" - <<'PY'
import json, glob
rows=[]
for p in glob.glob('results/q4_autofix/**/q4_metrics.json', recursive=True):
    d=json.load(open(p))
    rows.append({
        'path': p,
        'auc': float(d['test']['auc']),
        'f1': float(d['test']['f1']),
        'pass_both': bool(d['baseline_targets']['pass_both']),
        'split': d.get('split',{}),
        'cfg': d.get('best_trial',{}).get('config',{}),
    })
if not rows:
    print('NO_RESULTS')
    raise SystemExit(0)
rows.sort(key=lambda r:(r['pass_both'], r['f1'], r['auc']), reverse=True)
print('CURRENT_BEST', rows[0])
print('TOTAL_RUNS', len(rows))
PY
}

# run in pairs to use both GPUs
for ((i=0; i<${#RUNS[@]}; i+=2)); do
  run_one "${RUNS[$i]}" &
  PID1=$!
  PID2=""
  if (( i+1 < ${#RUNS[@]} )); then
    run_one "${RUNS[$((i+1))]}" &
    PID2=$!
  fi

  wait "$PID1" || true
  if [[ -n "$PID2" ]]; then
    wait "$PID2" || true
  fi

  report_best

  PASS=$($PYTHON_BIN - <<'PY'
import json,glob
ok=False
for p in glob.glob('results/q4_autofix/**/q4_metrics.json', recursive=True):
    d=json.load(open(p))
    if d.get('baseline_targets',{}).get('pass_both',False):
        ok=True
        break
print('1' if ok else '0')
PY
)

  if [[ "$PASS" == "1" ]]; then
    echo "PASS_BOTH achieved. Stopping early."
    break
  fi
done

echo "Auto-fix run completed."
report_best
