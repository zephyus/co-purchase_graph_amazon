import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from graph_models import BilinearDecoder, GATEncoder, LinkPredictionModel
from utils_graph import (
    binary_metrics,
    build_directed_edge_index,
    build_undirected_edge_set,
    choose_device,
    ensure_dir,
    explain_q3_choice_text,
    find_best_threshold_by_f1,
    load_dataset,
    roc_auc_binary,
    sample_negative_edges,
    save_json,
    set_seed,
    temporal_split_edges,
)


def edges_to_tensor(edges: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(edges).long().to(device)


def eval_split(
    model: torch.nn.Module,
    x: torch.Tensor,
    train_graph_edge_index: torch.Tensor,
    pos_edges: np.ndarray,
    neg_edges: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        pos_tensor = edges_to_tensor(pos_edges, device)
        neg_tensor = edges_to_tensor(neg_edges, device)

        pos_logits, _ = model(x, train_graph_edge_index, pos_tensor)
        neg_logits, _ = model(x, train_graph_edge_index, neg_tensor)

        y_true = np.concatenate(
            [
                np.ones(pos_edges.shape[0], dtype=np.int64),
                np.zeros(neg_edges.shape[0], dtype=np.int64),
            ]
        )
        y_prob = torch.sigmoid(torch.cat([pos_logits, neg_logits], dim=0)).detach().cpu().numpy()

        auc = roc_auc_binary(y_true, y_prob)
        loss = F.binary_cross_entropy(
            torch.from_numpy(y_prob).float(),
            torch.from_numpy(y_true).float(),
        ).item()

        return {
            "loss": float(loss),
            "auc": float(auc),
            "y_true": y_true,
            "y_prob": y_prob,
        }


def plot_curves(history: Dict[str, List[float]], out_path: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(epochs, history["train_loss"], label="train_loss")
    ax[0].plot(epochs, history["val_loss"], label="val_loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Link Prediction Loss")
    ax[0].legend()

    ax[1].plot(epochs, history["val_auc"], label="val_auc")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("AUC")
    ax[1].set_title("Validation AUC")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Q3: Future co-purchase link prediction")
    parser.add_argument("--dataset-dir", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="results/q3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)

    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)

    data = load_dataset(args.dataset_dir)
    num_nodes = data.x.size(0)
    x = data.x.to(device)

    train_pos, val_pos, test_pos = temporal_split_edges(
        data.edges_undirected,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    all_positive_set = build_undirected_edge_set(data.edges_undirected)
    val_neg = sample_negative_edges(
        num_nodes=num_nodes,
        num_samples=int(len(val_pos) * args.neg_ratio),
        positive_edge_set=all_positive_set,
        rng=rng,
    )
    test_neg = sample_negative_edges(
        num_nodes=num_nodes,
        num_samples=int(len(test_pos) * args.neg_ratio),
        positive_edge_set=all_positive_set,
        rng=rng,
    )

    train_graph_edge_index = build_directed_edge_index(train_pos, num_nodes, add_self_loops=True).to(device)

    encoder = GATEncoder(
        in_dim=x.size(1),
        hidden_dim=args.hidden_dim,
        out_dim=args.embed_dim,
        heads=args.heads,
        dropout=args.dropout,
    )
    decoder = BilinearDecoder(dim=args.embed_dim)
    model = LinkPredictionModel(encoder=encoder, decoder=decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
    }

    best_state = None
    best_epoch = -1
    best_val_auc = -1.0
    patience_counter = 0

    train_pos_tensor = edges_to_tensor(train_pos, device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Dynamic negative sampling for better training signal.
        train_neg = sample_negative_edges(
            num_nodes=num_nodes,
            num_samples=int(len(train_pos) * args.neg_ratio),
            positive_edge_set=all_positive_set,
            rng=rng,
        )
        train_neg_tensor = edges_to_tensor(train_neg, device)

        pos_logits, _ = model(x, train_graph_edge_index, train_pos_tensor)
        neg_logits, _ = model(x, train_graph_edge_index, train_neg_tensor)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat(
            [
                torch.ones_like(pos_logits),
                torch.zeros_like(neg_logits),
            ],
            dim=0,
        )

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        val_eval = eval_split(
            model=model,
            x=x,
            train_graph_edge_index=train_graph_edge_index,
            pos_edges=val_pos,
            neg_edges=val_neg,
            device=device,
        )

        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(val_eval["loss"])
        history["val_auc"].append(val_eval["auc"])

        if val_eval["auc"] > best_val_auc:
            best_val_auc = val_eval["auc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | train_loss={loss.item():.4f} "
                f"val_loss={val_eval['loss']:.4f} val_auc={val_eval['auc']:.4f}"
            )

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}, best epoch {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("No best model state saved")

    model.load_state_dict(best_state)

    val_eval = eval_split(
        model=model,
        x=x,
        train_graph_edge_index=train_graph_edge_index,
        pos_edges=val_pos,
        neg_edges=val_neg,
        device=device,
    )
    test_eval = eval_split(
        model=model,
        x=x,
        train_graph_edge_index=train_graph_edge_index,
        pos_edges=test_pos,
        neg_edges=test_neg,
        device=device,
    )

    best_threshold, val_thr_metrics = find_best_threshold_by_f1(val_eval["y_true"], val_eval["y_prob"])
    test_metrics = binary_metrics(test_eval["y_true"], test_eval["y_prob"], best_threshold)

    plot_curves(history, str(Path(args.output_dir) / "training_curves.png"))

    explain = explain_q3_choice_text(args.train_ratio, args.val_ratio, args.test_ratio)
    with open(Path(args.output_dir) / "q3_method_explanation.txt", "w", encoding="utf-8") as f:
        f.write(explain + "\n")

    results = {
        "seed": args.seed,
        "device": str(device),
        "split": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "train_positive_edges": int(len(train_pos)),
            "val_positive_edges": int(len(val_pos)),
            "test_positive_edges": int(len(test_pos)),
            "val_negative_edges": int(len(val_neg)),
            "test_negative_edges": int(len(test_neg)),
        },
        "loss_function": "BCEWithLogitsLoss",
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "threshold_selected_on_val": float(best_threshold),
        "val_metrics_at_selected_threshold": val_thr_metrics,
        "test": {
            "auc": float(test_eval["auc"]),
            "f1": float(test_metrics["f1"]),
            "precision": float(test_metrics["precision"]),
            "recall": float(test_metrics["recall"]),
            "accuracy": float(test_metrics["accuracy"]),
        },
    }

    save_json(results, str(Path(args.output_dir) / "q3_metrics.json"))
    torch.save(best_state, str(Path(args.output_dir) / "best_q3_model.pt"))

    print("Q3 completed")
    print(f"Best val AUC: {best_val_auc:.4f} (epoch {best_epoch})")
    print(f"Test AUC: {test_eval['auc']:.4f}, Test F1: {test_metrics['f1']:.4f}")
    print(f"Selected threshold: {best_threshold:.4f}")
    print(f"Artifacts saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
