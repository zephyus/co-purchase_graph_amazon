import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


@dataclass
class DatasetBundle:
    node_ids: np.ndarray
    x: torch.Tensor
    y: torch.Tensor
    edges_undirected: np.ndarray
    class_names: Dict[int, str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_device(preference: str = "auto", min_free_gb: float = 1.5) -> torch.device:
    pref = preference.lower()
    if pref not in {"auto", "cpu", "cuda"}:
        raise ValueError("preference must be one of: auto, cpu, cuda")

    if pref == "cpu":
        return torch.device("cpu")

    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")

    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes >= int(min_free_gb * (1024 ** 3)):
            return torch.device("cuda")
        return torch.device("cpu")
    except Exception:
        return torch.device("cuda")


def unique_undirected_edges_preserve_order(edge_pairs: np.ndarray) -> np.ndarray:
    seen: Set[Tuple[int, int]] = set()
    uniq: List[Tuple[int, int]] = []
    for u, v in edge_pairs.tolist():
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        key = (int(a), int(b))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    return np.asarray(uniq, dtype=np.int64)


def load_dataset(dataset_dir: str) -> DatasetBundle:
    dataset_path = Path(dataset_dir)
    nodes_df = pd.read_csv(dataset_path / "nodes.csv")
    edges_df = pd.read_csv(dataset_path / "edges.csv")
    classes_df = pd.read_csv(dataset_path / "classes.csv")

    required_node_cols = {"node_id", "label"}
    if not required_node_cols.issubset(nodes_df.columns):
        raise ValueError(f"nodes.csv must contain columns: {required_node_cols}")
    if not {"source", "target"}.issubset(edges_df.columns):
        raise ValueError("edges.csv must contain source,target")
    if not {"class_id", "class_name"}.issubset(classes_df.columns):
        raise ValueError("classes.csv must contain class_id,class_name")

    nodes_df = nodes_df.sort_values("node_id").reset_index(drop=True)
    node_ids = nodes_df["node_id"].to_numpy(dtype=np.int64)
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids.tolist())}

    feature_cols = [c for c in nodes_df.columns if c not in ("node_id", "label")]
    x_np = nodes_df[feature_cols].to_numpy(dtype=np.float32).copy()
    y_np = nodes_df["label"].to_numpy(dtype=np.int64).copy()

    raw_edges = edges_df[["source", "target"]].to_numpy(dtype=np.int64)
    mapped_edges: List[Tuple[int, int]] = []
    for s, t in raw_edges.tolist():
        if s not in node_id_to_idx or t not in node_id_to_idx:
            continue
        mapped_edges.append((node_id_to_idx[s], node_id_to_idx[t]))

    edges_undirected = unique_undirected_edges_preserve_order(np.asarray(mapped_edges, dtype=np.int64))

    class_names = {
        int(row.class_id): str(row.class_name)
        for row in classes_df.itertuples(index=False)
    }

    return DatasetBundle(
        node_ids=node_ids,
        x=torch.from_numpy(x_np),
        y=torch.from_numpy(y_np),
        edges_undirected=edges_undirected,
        class_names=class_names,
    )


def build_directed_edge_index(
    edges_undirected: np.ndarray,
    num_nodes: int,
    add_self_loops: bool = True,
) -> torch.Tensor:
    if edges_undirected.ndim != 2 or edges_undirected.shape[1] != 2:
        raise ValueError("edges_undirected must be shape [E, 2]")

    rev = edges_undirected[:, [1, 0]]
    directed = np.concatenate([edges_undirected, rev], axis=0)
    if add_self_loops:
        loops = np.arange(num_nodes, dtype=np.int64)
        loops = np.stack([loops, loops], axis=1)
        directed = np.concatenate([directed, loops], axis=0)

    directed_unique = np.unique(directed, axis=0)
    edge_index = torch.from_numpy(directed_unique.T).long()
    return edge_index


def stratified_split_indices(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    classes = np.unique(labels)
    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val

        train_idx.extend(cls_idx[:n_train].tolist())
        val_idx.extend(cls_idx[n_train:n_train + n_val].tolist())
        test_idx.extend(cls_idx[n_train + n_val:n_train + n_val + n_test].tolist())

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def mask_from_indices(num_nodes: int, indices: np.ndarray) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[torch.from_numpy(indices)] = True
    return mask


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]

    ranks = np.empty(len(y_score), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = ranks[y_true == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = y_true.astype(np.int64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def find_best_threshold_by_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_thresholds: int = 401,
) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metrics = binary_metrics(y_true, y_prob, best_threshold)
    for thr in np.linspace(0.0, 1.0, num_thresholds):
        m = binary_metrics(y_true, y_prob, float(thr))
        if m["f1"] > best_metrics["f1"]:
            best_threshold = float(thr)
            best_metrics = m
    return best_threshold, best_metrics


def pca_project_2d(x: np.ndarray) -> np.ndarray:
    x_centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    proj = x_centered @ vt[:2].T
    return proj


def plot_embedding(
    emb_2d: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    title: str,
    class_names: Optional[Dict[int, str]] = None,
) -> None:
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = class_names.get(int(label), str(label)) if class_names else str(label)
        plt.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            s=8,
            alpha=0.7,
            label=label_name,
            color=cmap(i),
        )

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=2.0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_json(data: Dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_undirected_edge_set(edges: np.ndarray) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for u, v in edges.tolist():
        a, b = (int(u), int(v)) if u < v else (int(v), int(u))
        if a != b:
            out.add((a, b))
    return out


def temporal_split_edges(
    edges_undirected_in_order: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    n = edges_undirected_in_order.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_pos = edges_undirected_in_order[:n_train]
    val_pos = edges_undirected_in_order[n_train:n_train + n_val]
    test_pos = edges_undirected_in_order[n_train + n_val:n_train + n_val + n_test]
    return train_pos, val_pos, test_pos


def sample_negative_edges(
    num_nodes: int,
    num_samples: int,
    positive_edge_set: Set[Tuple[int, int]],
    rng: np.random.Generator,
    forbidden: Optional[Set[Tuple[int, int]]] = None,
) -> np.ndarray:
    chosen: Set[Tuple[int, int]] = set()
    if forbidden is None:
        forbidden = set()

    max_attempts = max(50_000, num_samples * 25)
    attempts = 0
    while len(chosen) < num_samples and attempts < max_attempts:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        attempts += 1
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if key in positive_edge_set or key in forbidden or key in chosen:
            continue
        chosen.add(key)

    if len(chosen) < num_samples:
        needed = num_samples - len(chosen)
        for u in range(num_nodes):
            if needed <= 0:
                break
            v = (u + 1) % num_nodes
            a, b = (u, v) if u < v else (v, u)
            key = (a, b)
            if key in positive_edge_set or key in forbidden or key in chosen or a == b:
                continue
            chosen.add(key)
            needed -= 1

    if len(chosen) < num_samples:
        raise RuntimeError("Unable to sample enough negative edges")

    return np.asarray(list(chosen), dtype=np.int64)


def build_neighbor_sets(num_nodes: int, edges_undirected: np.ndarray) -> List[Set[int]]:
    neighbors: List[Set[int]] = [set() for _ in range(num_nodes)]
    for u, v in edges_undirected.tolist():
        neighbors[int(u)].add(int(v))
        neighbors[int(v)].add(int(u))
    return neighbors


def sample_hard_negative_edges(
    num_nodes: int,
    num_samples: int,
    positive_edge_set: Set[Tuple[int, int]],
    neighbors: List[Set[int]],
    rng: np.random.Generator,
    hard_fraction: float = 0.5,
) -> np.ndarray:
    hard_target = int(num_samples * hard_fraction)
    chosen: Set[Tuple[int, int]] = set()

    nodes_with_neighbors = [i for i in range(num_nodes) if neighbors[i]]
    attempts = 0
    max_attempts = max(100_000, hard_target * 40)

    while len(chosen) < hard_target and attempts < max_attempts and nodes_with_neighbors:
        u = int(nodes_with_neighbors[int(rng.integers(0, len(nodes_with_neighbors)))])
        u_nei = list(neighbors[u])
        mid = int(u_nei[int(rng.integers(0, len(u_nei)))])
        second = neighbors[mid]
        if not second:
            attempts += 1
            continue
        v = int(list(second)[int(rng.integers(0, len(second)))])
        attempts += 1
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if key in positive_edge_set or key in chosen:
            continue
        chosen.add(key)

    needed_random = num_samples - len(chosen)
    random_part = sample_negative_edges(
        num_nodes=num_nodes,
        num_samples=needed_random,
        positive_edge_set=positive_edge_set,
        rng=rng,
        forbidden=chosen,
    )

    out = list(chosen)
    out.extend(map(tuple, random_part.tolist()))
    return np.asarray(out, dtype=np.int64)


def explain_q3_choice_text(train_ratio: float, val_ratio: float, test_ratio: float) -> str:
    return (
        "Future co-purchase is approximated with edge row order as pseudo-time. "
        f"Positive edges are split chronologically with ratio {train_ratio:.2f}/{val_ratio:.2f}/{test_ratio:.2f} "
        "for train/validation/test, so training only sees earlier edges and evaluation is on later edges. "
        "BCEWithLogitsLoss is used because link prediction is binary and this loss is numerically stable for logits-based training."
    )
