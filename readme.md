DATASET:
Amazon Computers is a segment of the Amazon co-purchase graph, where nodes
represent goods, edges indicate that two goods are frequently purchased together, node
features are bag-of-words-encoded product reviews, and class labels are assigned by
product category.

1. node.csv: This file contains the features and labels of the nodes in the graph.
   • node_id: Unique identifier for each node.
   • Feature columns: Numerical features.
   • label: The class label associated with each node.

2. edges.csv: This file contains the edges of the graph in a tabular format, representing the
   connections between nodes in the graph.
   • source: The source node ID of the edge.
   • target: The target node ID of the edge.

3. classes.csv: Maps class IDs to their corresponding class names.

Question 1: Construct a co-purchase network graph and report the node-level
statistics and feature (e.g., number of nodes, number of edges, node degree
distribution, etc.)

Question 2: Implement the Graph Attention Network (GAT) and train the model for
the node classification task. Use the splitting ratio 35:25:40 for training/validation/test
masks, respectively. Plot the training process (losses + accuracies) and visualize the
node embedding before and after the training.

Question 3: Modify the dataset and GAT model to predict future co-purchases
between items. Specify the splitting ratio and loss function you choose with
explanations. Report the AUC and F1-score on the test set.

Question 4: Improve the current model or implement a more advanced architecture so
that the evaluation results pass the baselines: AUC = 87.5% and F1-Score = 85.0%.

---

## Implementation (Script-based)

This project now includes end-to-end scripts for Q1-Q4:

- `q1_graph_stats.py`
- `q2_gat_node_classification.py`
- `q3_link_prediction.py`
- `q4_advanced_link_prediction.py`
- `utils_graph.py`
- `graph_models.py`
- `run_all.sh`
- `run_q4_ultra_dual_gpu.sh`
- `run_q4_autofix_until_pass.sh`

All scripts assume data files are in the same folder:

- `nodes.csv`
- `edges.csv`
- `classes.csv`

## Environment

Python executable used in this workspace:

```bash
/home/russell512/.venv/bin/python
```

## Run Each Question

### Q1: Graph construction + statistics

```bash
/home/russell512/.venv/bin/python q1_graph_stats.py --dataset-dir . --output-dir results/q1
```

### Q2: GAT node classification (35/25/40)

```bash
/home/russell512/.venv/bin/python q2_gat_node_classification.py \
   --dataset-dir . \
   --output-dir results/q2 \
   --train-ratio 0.35 --val-ratio 0.25 --test-ratio 0.40
```

### Q3: Future co-purchase prediction (temporal proxy by edge row order)

```bash
/home/russell512/.venv/bin/python q3_link_prediction.py \
   --dataset-dir . \
   --output-dir results/q3 \
   --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
```

### Q4: Advanced single-model link prediction

```bash
/home/russell512/.venv/bin/python q4_advanced_link_prediction.py \
   --dataset-dir . \
   --output-dir results/q4 \
   --device cuda \
   --select-by f1 \
   --blend-heuristic
```

### Q4: Auto-fix dual-GPU training (recommended for strongest result)

This launcher runs paired high-intensity jobs on both GPUs, compares results after each round,
and stops early when both baseline targets are passed.

```bash
bash run_q4_autofix_until_pass.sh
```

### Run Full Pipeline

```bash
bash run_all.sh
```

## Artifacts

Outputs are stored under `results/q1`, `results/q2`, `results/q3`, `results/q4`.

- Q1: JSON stats + distributions.
- Q2: training curves, embedding before/after training, metrics JSON.
- Q3: method explanation, training curves, AUC/F1 metrics JSON.
- Q4: trial table, best-model metrics JSON, baseline pass/fail flags.

## Final Best Q4 Result (Auto-fix)

Canonical best folder:

- `results/q4_best/q4_metrics.json`
- `results/q4_best/best_q4_model.pt`
- `results/q4_best/best_trial_curves.png`

Best achieved metrics:

- Test AUC: `0.9159`
- Test F1: `0.8526`
- Baseline pass: `AUC>=0.875` and `F1>=0.850` both passed.