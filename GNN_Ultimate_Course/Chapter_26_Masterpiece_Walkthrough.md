# Chapter 26: Amazon Graph Masterpiece Walkthrough

## 1. The Intuition (引言與核心靈魂)
Imagine trying to predict whether two completely strangers on opposite sides of the world will become best friends. It’s impossible if you only know what they look like. But if you see they both frequent the same underground jazz club, buy the same obscure vinyl records, and share five mutual friends, the prediction becomes trivial.

In Chapter 19, we did link prediction the old way. In Chapter 23-25, we learned GNNs. Now, it is time to face the real world: The Amazon Co-Purchase Graph. We have millions of products (nodes) and their "frequently bought together" patterns (edges). This chapter walks through `q4_advanced_link_prediction.py`, an architecture designed precisely for industrial-scale link prediction.

**Learning Objectives:**
1. Deconstruct an advanced link prediction PyTorch pipeline.
2. Understand negative sampling strategies in massive, sparse graphs.
3. Master the art of modular GNN architecture design.

## 2. Deep Dive (核心概念與深度解析)
Link Prediction is formulated as a binary classification problem over pairs of nodes $(u, v)$. 
Given a graph $G = (V, E)$, our goal is to learn a mapping $f: V \times V \rightarrow [0, 1]$ representing the probability that $(u, v) \in E$.

The architecture in `q4_advanced_link_prediction.py` follows an Encoder-Decoder paradigm:
1. **The Encoder (GNN):** Maps node $i$ to a high-dimensional continuous vector $\mathbf{z}_i \in \mathbb{R}^d$.
   $$ \mathbf{Z} = \text{Encoder}(X, A) $$
2. **The Decoder (Edge Predictor):** Takes two node embeddings $\mathbf{z}_u$ and $\mathbf{z}_v$ and outputs a scalar probability. The dot-product decoder is classic:
   $$ \hat{y}_{uv} = \sigma(\mathbf{z}_u^T \mathbf{z}_v) $$
Alternatively, modern architectures concatenate the embeddings and pass them through an MLP:
   $$ \hat{y}_{uv} = \sigma(\text{MLP}([\mathbf{z}_u || \mathbf{z}_v || (\mathbf{z}_u \odot \mathbf{z}_v)])) $$
   Where $\odot$ represents the Hadamard (element-wise) product, which preserves symmetric interaction features perfectly.

**Negative Sampling Ratio:**
In Amazon's graph, existing links are astronomically outnumbered by non-existing links. If we use all negative edges, we suffer from extreme class imbalance. The script dynamically samples negative edges at a $1:1$ or $1:3$ ratio during training to maintain a balanced gradient landscape.

**Common Misconceptions:**
- *We should train the GNN on all edges.* False. If you train the GNN using the edge $(u, v)$ and then ask it to predict $(u, v)$, you commit data leakage. Target edges must be strictly removed from the message-passing graph (the `edge_index`).

## 3. Code & Engineering (程式碼實作與工程解密)
```python
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class AdvancedLinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        # Encoder: 2-layer GraphSAGE
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Decoder: MLP for robust interaction learning
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Message passing generates node embeddings
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        # Extract embeddings for the target edge pairs
        src, dst = edge_label_index
        z_src = z[src]
        z_dst = z[dst]
        
        # Create rich edge representation
        edge_feat = torch.cat([
            z_src, 
            z_dst, 
            z_src * z_dst  # Hadamard product interacts features directly
        ], dim=-1)
        
        # Output raw logits (no sigmoid here if using BCEWithLogitsLoss)
        return self.predictor(edge_feat).squeeze()
```

*Engineering Note:* Notice how the `encode` and `decode` steps are cleanly separated. During inference in production, Amazon would precompute all $\mathbf{Z}$ embeddings overnight. When a user views item $A$, the system only runs the fast O(1) `decode` function against candidate items, rather than re-running the heavy GNN.

## 4. MIT-Level Exercises (課後思考與魔王挑戰)
1. **Conceptual Validation:** Explain why the Hadamard product ($\mathbf{z}_u \odot \mathbf{z}_v$) provides permutation invariance ($f(u,v) = f(v,u)$), but the concatenation $[\mathbf{z}_u || \mathbf{z}_v]$ breaks it. How does the MLP layer compensate for this?
2. **Extreme Edge-Case:** You execute this code on a graph with 500 million edges. The system crashes with a CUDA Out-of-Memory error exclusively during the `torch.cat` operation in the `decode` phase. How would you mathematically refactor the `decode` step to compute predictions in chunks without breaking autograd?