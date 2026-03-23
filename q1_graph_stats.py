import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils_graph import ensure_dir, load_dataset, save_json


def connected_component_sizes(num_nodes: int, edges: np.ndarray) -> np.ndarray:
    parent = np.arange(num_nodes, dtype=np.int64)
    rank = np.zeros(num_nodes, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for u, v in edges.tolist():
        union(int(u), int(v))

    roots = np.array([find(i) for i in range(num_nodes)], dtype=np.int64)
    _, counts = np.unique(roots, return_counts=True)
    counts.sort()
    return counts[::-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Q1: Co-purchase graph statistics")
    parser.add_argument("--dataset-dir", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="results/q1")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    data = load_dataset(args.dataset_dir)
    n_nodes = data.x.size(0)
    n_features = data.x.size(1)
    n_edges = int(data.edges_undirected.shape[0])

    degrees = np.zeros(n_nodes, dtype=np.int64)
    for u, v in data.edges_undirected.tolist():
        degrees[int(u)] += 1
        degrees[int(v)] += 1

    density = (2.0 * n_edges) / (n_nodes * (n_nodes - 1))
    isolated_nodes = int((degrees == 0).sum())

    x_np = data.x.numpy()
    nonzero_per_node = (x_np != 0).sum(axis=1)
    feature_density_per_node = nonzero_per_node / n_features
    global_feature_density = float((x_np != 0).sum() / x_np.size)

    y_np = data.y.numpy()
    unique_labels, counts = np.unique(y_np, return_counts=True)
    label_distribution = {
        int(lbl): {
            "count": int(cnt),
            "class_name": data.class_names.get(int(lbl), str(lbl)),
        }
        for lbl, cnt in zip(unique_labels.tolist(), counts.tolist())
    }

    connected_components = connected_component_sizes(n_nodes, data.edges_undirected)

    stats = {
        "num_nodes": int(n_nodes),
        "num_edges": int(n_edges),
        "num_features": int(n_features),
        "num_classes": int(len(unique_labels)),
        "graph_density": float(density),
        "isolated_nodes": int(isolated_nodes),
        "degree": {
            "min": int(degrees.min()),
            "max": int(degrees.max()),
            "mean": float(degrees.mean()),
            "median": float(np.median(degrees)),
            "std": float(degrees.std()),
            "p95": float(np.percentile(degrees, 95)),
            "p99": float(np.percentile(degrees, 99)),
        },
        "feature_sparsity": {
            "global_density": global_feature_density,
            "global_sparsity": float(1.0 - global_feature_density),
            "nonzero_features_per_node_mean": float(nonzero_per_node.mean()),
            "nonzero_features_per_node_median": float(np.median(nonzero_per_node)),
            "nonzero_features_per_node_std": float(nonzero_per_node.std()),
        },
        "connected_components": {
            "count": int(connected_components.shape[0]),
            "largest_component_size": int(connected_components[0]),
            "smallest_component_size": int(connected_components[-1]),
            "top_10_sizes": [int(v) for v in connected_components[:10].tolist()],
        },
        "label_distribution": label_distribution,
    }

    out_json = str(Path(args.output_dir) / "q1_stats.json")
    save_json(stats, out_json)

    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=60, color="#1f77b4", alpha=0.85)
    plt.title("Degree Distribution")
    plt.xlabel("Node degree")
    plt.ylabel("Node count")
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "degree_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(feature_density_per_node, bins=60, color="#ff7f0e", alpha=0.85)
    plt.title("Feature Density per Node")
    plt.xlabel("Non-zero feature ratio")
    plt.ylabel("Node count")
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "feature_density_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 5))
    x_ticks = [data.class_names.get(int(lbl), str(lbl)) for lbl in unique_labels]
    plt.bar(x_ticks, counts, color="#2ca02c", alpha=0.85)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Node count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "class_distribution.png", dpi=200)
    plt.close()

    print("Q1 completed")
    print(f"Nodes: {n_nodes}, Edges: {n_edges}, Features: {n_features}")
    print(f"Average degree: {degrees.mean():.4f}, Density: {density:.8f}")
    print(f"Statistics saved to: {out_json}")


if __name__ == "__main__":
    main()
