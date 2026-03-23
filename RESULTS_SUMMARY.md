# Amazon Co-Purchase Graph - Final Results Summary

This file summarizes the final outputs generated in this repository for Q1-Q4.

## Q1. Co-purchase graph construction and statistics

Source output: results/q1/q1_stats.json

- Nodes: 13,752
- Undirected edges: 245,861
- Features per node: 767
- Classes: 10
- Graph density: 0.0026002763
- Isolated nodes: 281
- Degree mean/median/max: 35.7564 / 22 / 2992
- Connected components: 314
- Largest connected component: 13,381

Artifacts:
- results/q1/q1_stats.json
- results/q1/degree_distribution.png
- results/q1/feature_density_distribution.png
- results/q1/class_distribution.png

## Q2. GAT node classification (35:25:40)

Source output: results/q2/q2_metrics.json

- Split ratio (train/val/test): 0.35 / 0.25 / 0.40
- Device used: CUDA
- Best validation accuracy: 0.8965
- Test accuracy: 0.8900
- Test loss: 0.3419

Artifacts:
- results/q2/q2_metrics.json
- results/q2/training_curves.png
- results/q2/embedding_before_training.png
- results/q2/embedding_after_training.png
- results/q2/best_q2_model.pt

## Q3. Future co-purchase link prediction (temporal proxy)

Source output: results/q3/q3_metrics.json

Method:
- Future definition: edge row order in edges.csv as pseudo-time.
- Chronological split ratio (train/val/test): 0.70 / 0.15 / 0.15
- Loss: BCEWithLogitsLoss

Final metrics:
- Test AUC: 0.6668
- Test F1: 0.6667

Artifacts:
- results/q3/q3_metrics.json
- results/q3/q3_method_explanation.txt
- results/q3/training_curves.png
- results/q3/best_q3_model.pt

## Q4. Advanced single-model architecture

Best final run output: results/q4_final_push/q4_metrics.json

Model summary:
- Residual GAT encoder + MLP edge decoder
- Hard-negative-aware training
- Validation-threshold selection for F1

Best final metrics:
- Test AUC: 0.9119
- Test F1: 0.8431

Baseline check:
- AUC >= 0.875: PASS
- F1 >= 0.850: NOT PASS (close)

Artifacts:
- results/q4_final_push/q4_metrics.json
- results/q4_final_push/best_trial_curves.png
- results/q4_final_push/best_q4_model.pt

## Notes

- Multiple additional Q4 tuning runs are preserved under results/ for reproducibility.
- The strongest observed configuration reliably exceeds the AUC baseline and approaches the F1 target.
