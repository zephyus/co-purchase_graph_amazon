from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseGATLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.3,
        negative_slope: float = 0.2,
        concat: bool = True,
        edge_chunk_size: int = 100_000,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.edge_chunk_size = edge_chunk_size

        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def _segment_softmax(self, scores: torch.Tensor, index: torch.Tensor, n_nodes: int) -> torch.Tensor:
        # scores: [E, H], index: [E]
        h = scores.size(1)
        index_exp = index.unsqueeze(-1).expand(-1, h)

        max_per_node = torch.full(
            (n_nodes, h),
            fill_value=-1.0e15,
            device=scores.device,
            dtype=scores.dtype,
        )
        max_per_node.scatter_reduce_(0, index_exp, scores, reduce="amax", include_self=True)

        norm_scores = scores - max_per_node[index]
        exp_scores = torch.exp(norm_scores)

        denom = torch.zeros((n_nodes, h), device=scores.device, dtype=scores.dtype)
        denom.scatter_add_(0, index_exp, exp_scores)

        alpha = exp_scores / (denom[index] + 1e-12)
        return alpha

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n_nodes = x.size(0)
        src = edge_index[0]
        dst = edge_index[1]
        n_edges = src.size(0)

        x_proj = self.lin(x).view(n_nodes, self.heads, self.out_dim)

        # Compute attention scores in chunks to reduce peak memory on large graphs.
        att = torch.empty((n_edges, self.heads), device=x.device, dtype=x.dtype)
        for start in range(0, n_edges, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, n_edges)
            s = src[start:end]
            d = dst[start:end]
            x_src = x_proj[s]
            x_dst = x_proj[d]
            att_chunk = (x_src * self.att_src).sum(dim=-1) + (x_dst * self.att_dst).sum(dim=-1)
            att[start:end] = F.leaky_relu(att_chunk, negative_slope=self.negative_slope)

        alpha = self._segment_softmax(att, dst, n_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros(
            (n_nodes, self.heads, self.out_dim),
            device=x.device,
            dtype=x.dtype,
        )

        for start in range(0, n_edges, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, n_edges)
            s = src[start:end]
            d = dst[start:end]
            x_src = x_proj[s]
            alpha_chunk = alpha[start:end]
            messages = x_src * alpha_chunk.unsqueeze(-1)
            out.scatter_add_(0, d.view(-1, 1, 1).expand(-1, self.heads, self.out_dim), messages)

        if self.concat:
            out = out.reshape(n_nodes, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)

        out = out + self.bias
        return out


class GATNodeClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.gat1 = SparseGATLayer(in_dim=in_dim, out_dim=hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2 = SparseGATLayer(
            in_dim=hidden_dim * heads,
            out_dim=out_dim,
            heads=1,
            dropout=dropout,
            concat=False,
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.gat2(h, edge_index)
        return logits


class GATEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.gat1 = SparseGATLayer(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2 = SparseGATLayer(hidden_dim * heads, out_dim, heads=1, dropout=dropout, concat=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.gat2(h, edge_index)
        return z


class BilinearDecoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        logits = self.bilinear(z_src, z_dst).squeeze(-1)
        return logits


class ResidualGATBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layer = SparseGATLayer(in_dim=in_dim, out_dim=out_dim, heads=heads, dropout=dropout, concat=True)
        self.norm = nn.LayerNorm(out_dim * heads)
        self.dropout = dropout
        self.res_proj = None
        if in_dim != out_dim * heads:
            self.res_proj = nn.Linear(in_dim, out_dim * heads)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.layer(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        residual = self.res_proj(x) if self.res_proj is not None else x
        out = self.norm(h + residual)
        return out


class AdvancedGATEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block1 = ResidualGATBlock(in_dim=in_dim, out_dim=hidden_dim, heads=heads, dropout=dropout)
        self.block2 = ResidualGATBlock(
            in_dim=hidden_dim * heads,
            out_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
        )
        self.proj = nn.Linear(hidden_dim * heads, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.block1(x, edge_index)
        h = self.block2(h, edge_index)
        z = self.proj(h)
        return z


class EdgeMLPDecoder(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z_src, z_dst, z_src * z_dst, torch.abs(z_src - z_dst)], dim=-1)
        logits = self.mlp(h).squeeze(-1)
        return logits


class LinkPredictionModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        edge_batch_size: int = 100_000,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.edge_batch_size = edge_batch_size

    def forward(
        self,
        x: torch.Tensor,
        edge_index_graph: torch.Tensor,
        edge_pairs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x, edge_index_graph)
        logits_chunks = []
        for start in range(0, edge_pairs.size(0), self.edge_batch_size):
            end = min(start + self.edge_batch_size, edge_pairs.size(0))
            batch = edge_pairs[start:end]
            src = batch[:, 0]
            dst = batch[:, 1]
            logits_chunks.append(self.decoder(z[src], z[dst]))
        logits = torch.cat(logits_chunks, dim=0)
        return logits, z
