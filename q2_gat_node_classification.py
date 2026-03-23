import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from graph_models import GATNodeClassifier
from utils_graph import (
    accuracy_from_logits,
    build_directed_edge_index,
    choose_device,
    ensure_dir,
    load_dataset,
    mask_from_indices,
    pca_project_2d,
    plot_embedding,
    save_json,
    set_seed,
    stratified_split_indices,
)


def evaluate(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[mask], y[mask])
        acc = accuracy_from_logits(logits[mask], y[mask])
    return {"loss": float(loss.item()), "acc": float(acc)}


def save_training_plot(history: Dict[str, List[float]], out_path: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(epochs, history["train_loss"], label="train_loss")
    ax[0].plot(epochs, history["val_loss"], label="val_loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Training / Validation Loss")
    ax[0].legend()

    ax[1].plot(epochs, history["train_acc"], label="train_acc")
    ax[1].plot(epochs, history["val_acc"], label="val_acc")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Training / Validation Accuracy")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2: GAT node classification")
    parser.add_argument("--dataset-dir", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="results/q2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.35)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--test-ratio", type=float, default=0.40)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)

    device = choose_device(args.device)
    data = load_dataset(args.dataset_dir)

    x = data.x.to(device)
    y = data.y.to(device)
    num_nodes = x.size(0)
    num_classes = int(y.max().item()) + 1

    edge_index = build_directed_edge_index(data.edges_undirected, num_nodes, add_self_loops=True).to(device)

    y_np = data.y.numpy()
    train_idx, val_idx, test_idx = stratified_split_indices(
        labels=y_np,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_mask = mask_from_indices(num_nodes, train_idx).to(device)
    val_mask = mask_from_indices(num_nodes, val_idx).to(device)
    test_mask = mask_from_indices(num_nodes, test_idx).to(device)

    model = GATNodeClassifier(
        in_dim=x.size(1),
        hidden_dim=args.hidden_dim,
        out_dim=num_classes,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Embedding before training.
    model.eval()
    with torch.no_grad():
        emb_before = model.encode(x, edge_index).detach().cpu().numpy()
    emb_before_2d = pca_project_2d(emb_before)
    plot_embedding(
        emb_2d=emb_before_2d,
        labels=y_np,
        out_path=str(Path(args.output_dir) / "embedding_before_training.png"),
        title="Node Embedding Before Training (PCA)",
        class_names=data.class_names,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        train_acc = accuracy_from_logits(logits[train_mask], y[train_mask])
        val_metrics = evaluate(model, x, y, edge_index, val_mask)

        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(val_metrics["acc"])

        improved = val_metrics["acc"] > best_val_acc
        if improved:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={loss.item():.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
            )

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}, best epoch {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("No best model state saved")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(x, edge_index)
        test_loss = F.cross_entropy(logits[test_mask], y[test_mask]).item()
        test_acc = accuracy_from_logits(logits[test_mask], y[test_mask])

        pred = logits.argmax(dim=-1)
        emb_after = model.encode(x, edge_index).detach().cpu().numpy()

    emb_after_2d = pca_project_2d(emb_after)
    plot_embedding(
        emb_2d=emb_after_2d,
        labels=y_np,
        out_path=str(Path(args.output_dir) / "embedding_after_training.png"),
        title="Node Embedding After Training (PCA)",
        class_names=data.class_names,
    )

    save_training_plot(history, str(Path(args.output_dir) / "training_curves.png"))

    num_correct_per_class = {}
    y_cpu = y.detach().cpu().numpy()
    pred_cpu = pred.detach().cpu().numpy()
    for cls in np.unique(y_cpu):
        mask = y_cpu == cls
        acc_cls = float((pred_cpu[mask] == y_cpu[mask]).mean())
        num_correct_per_class[int(cls)] = {
            "accuracy": acc_cls,
            "class_name": data.class_names.get(int(cls), str(int(cls))),
            "count": int(mask.sum()),
        }

    results = {
        "seed": args.seed,
        "device": str(device),
        "split": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "train_count": int(train_mask.sum().item()),
            "val_count": int(val_mask.sum().item()),
            "test_count": int(test_mask.sum().item()),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "hidden_dim": args.hidden_dim,
            "heads": args.heads,
        },
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "per_class": num_correct_per_class,
    }

    save_json(results, str(Path(args.output_dir) / "q2_metrics.json"))
    torch.save(best_state, str(Path(args.output_dir) / "best_q2_model.pt"))

    print("Q2 completed")
    print(f"Best val acc: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Test acc: {test_acc:.4f}, Test loss: {test_loss:.4f}")
    print(f"Artifacts saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
