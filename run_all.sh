#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/russell512/.venv/bin/python"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT_DIR"

echo "[1/4] Q1 graph statistics"
"$PYTHON_BIN" q1_graph_stats.py --dataset-dir "$ROOT_DIR" --output-dir "$ROOT_DIR/results/q1"

echo "[2/4] Q2 GAT node classification"
"$PYTHON_BIN" q2_gat_node_classification.py \
  --dataset-dir "$ROOT_DIR" \
  --output-dir "$ROOT_DIR/results/q2" \
  --epochs 300 \
  --patience 40

echo "[3/4] Q3 future link prediction"
"$PYTHON_BIN" q3_link_prediction.py \
  --dataset-dir "$ROOT_DIR" \
  --output-dir "$ROOT_DIR/results/q3" \
  --epochs 220 \
  --patience 30

echo "[4/4] Q4 advanced link prediction"
"$PYTHON_BIN" q4_advanced_link_prediction.py \
  --dataset-dir "$ROOT_DIR" \
  --output-dir "$ROOT_DIR/results/q4" \
  --epochs 180 \
  --patience 25 \
  --device cpu

echo "All tasks finished. Check $ROOT_DIR/results"
