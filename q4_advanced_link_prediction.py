import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from graph_models import AdvancedGATEncoder, EdgeMLPDecoder, LinkPredictionModel
from utils_graph import (
    binary_metrics,
    build_directed_edge_index,
    build_neighbor_sets,
    build_undirected_edge_set,
    choose_device,
    ensure_dir,
    find_best_threshold_by_f1,
    load_dataset,
    roc_auc_binary,
    sample_hard_negative_edges,
    sample_negative_edges,
    save_json,
    set_seed,
    temporal_split_edges,
)


@dataclass
class TrialConfig:
    hidden_dim: int
    heads: int
    dropout: float
    embed_dim: int
    decoder_hidden_dim: int
    lr: float
    weight_decay: float
    neg_ratio: float
    hard_fraction: float


DEFAULT_TRIALS = [
    TrialConfig(24, 2, 0.30, 64, 96, 0.0020, 5e-5, 1.0, 0.40),
    TrialConfig(32, 2, 0.30, 64, 96, 0.0015, 1e-4, 1.0, 0.50),
    TrialConfig(24, 4, 0.35, 80, 128, 0.0015, 1e-4, 1.2, 0.50),
    TrialConfig(32, 4, 0.35, 96, 128, 0.0010, 1e-4, 1.2, 0.60),
    TrialConfig(32, 4, 0.25, 96, 160, 0.0012, 5e-5, 1.0, 0.45),
    TrialConfig(40, 4, 0.25, 96, 192, 0.0009, 5e-5, 1.0, 0.40),
    TrialConfig(48, 2, 0.20, 128, 192, 0.0007, 1e-4, 1.2, 0.60),
]


def edges_to_tensor(edges: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(edges).long().to(device)


def evaluate_split(
    model: torch.nn.Module,
    x: torch.Tensor,
    graph_edge_index: torch.Tensor,
    pos_edges: np.ndarray,
    neg_edges: np.ndarray,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        pos_tensor = edges_to_tensor(pos_edges, device)
        neg_tensor = edges_to_tensor(neg_edges, device)

        pos_logits, _ = model(x, graph_edge_index, pos_tensor)
        neg_logits, _ = model(x, graph_edge_index, neg_tensor)

        y_true = np.concatenate(
            [
                np.ones(len(pos_edges), dtype=np.int64),
                np.zeros(len(neg_edges), dtype=np.int64),
            ]
        )
        y_prob = torch.sigmoid(torch.cat([pos_logits, neg_logits], dim=0)).detach().cpu().numpy()
        auc = roc_auc_binary(y_true, y_prob)
    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "auc": float(auc),
    }


def train_one_trial(
    trial_id: int,
    cfg: TrialConfig,
    x: torch.Tensor,
    train_graph_edge_index: torch.Tensor,
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    val_neg: np.ndarray,
    all_positive_set,
    neighbors,
    device: torch.device,
    epochs: int,
    patience: int,
    seed: int,
    train_pos_sample_ratio: float,
) -> Dict:
    torch.manual_seed(seed + trial_id)

    encoder = AdvancedGATEncoder(
        in_dim=x.size(1),
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.embed_dim,
        heads=cfg.heads,
        dropout=cfg.dropout,
    )
    decoder = EdgeMLPDecoder(dim=cfg.embed_dim, hidden_dim=cfg.decoder_hidden_dim, dropout=cfg.dropout)
    model = LinkPredictionModel(encoder=encoder, decoder=decoder).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 10))

    criterion = torch.nn.BCEWithLogitsLoss()

    rng = np.random.default_rng(seed + 1000 + trial_id)
    best_state = None
    best_epoch = -1
    best_val_auc = -1.0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_auc": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        n_train_pos = max(1024, int(len(train_pos) * train_pos_sample_ratio))
        if n_train_pos >= len(train_pos):
            train_pos_batch = train_pos
        else:
            pos_idx = rng.choice(len(train_pos), size=n_train_pos, replace=False)
            train_pos_batch = train_pos[pos_idx]

        train_pos_tensor = edges_to_tensor(train_pos_batch, device)

        n_train_neg = max(1024, int(len(train_pos_batch) * cfg.neg_ratio))
        train_neg = sample_hard_negative_edges(
            num_nodes=x.size(0),
            num_samples=n_train_neg,
            positive_edge_set=all_positive_set,
            neighbors=neighbors,
            rng=rng,
            hard_fraction=cfg.hard_fraction,
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        val_eval = evaluate_split(
            model=model,
            x=x,
            graph_edge_index=train_graph_edge_index,
            pos_edges=val_pos,
            neg_edges=val_neg,
            device=device,
        )

        history["train_loss"].append(float(loss.item()))
        history["val_auc"].append(float(val_eval["auc"]))

        if val_eval["auc"] > best_val_auc:
            best_val_auc = float(val_eval["auc"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Trial {trial_id} | Epoch {epoch:03d} | "
                f"train_loss={loss.item():.4f} val_auc={val_eval['auc']:.4f}"
            )

        if patience_counter >= patience:
            print(f"Trial {trial_id} early stop at epoch {epoch}, best={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"Trial {trial_id}: no best model")

    return {
        "trial_id": trial_id,
        "config": cfg,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "history": history,
    }


def plot_best_trial_curve(history: Dict[str, List[float]], out_path: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(epochs, history["train_loss"], label="train_loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Best Trial Train Loss")
    ax[0].legend()

    ax[1].plot(epochs, history["val_auc"], label="val_auc")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("AUC")
    ax[1].set_title("Best Trial Validation AUC")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Q4: Advanced single-model link prediction")
    parser.add_argument("--dataset-dir", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="results/q4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="Run a smaller search for smoke testing")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--train-pos-sample-ratio", type=float, default=0.30)
    parser.add_argument(
        "--single-trial-index",
        type=int,
        default=0,
        help="1-based index into DEFAULT_TRIALS to run only one config (0 means normal multi-trial search)",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)

    device = choose_device(args.device, min_free_gb=2.0)
    data = load_dataset(args.dataset_dir)
    x = data.x.to(device)
    num_nodes = x.size(0)

    train_pos, val_pos, test_pos = temporal_split_edges(
        data.edges_undirected,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    all_positive_set = build_undirected_edge_set(data.edges_undirected)
    neighbors = build_neighbor_sets(num_nodes=num_nodes, edges_undirected=train_pos)

    rng = np.random.default_rng(args.seed)
    val_neg = sample_negative_edges(
        num_nodes=num_nodes,
        num_samples=len(val_pos),
        positive_edge_set=all_positive_set,
        rng=rng,
    )
    test_neg = sample_negative_edges(
        num_nodes=num_nodes,
        num_samples=len(test_pos),
        positive_edge_set=all_positive_set,
        rng=rng,
    )

    train_graph_edge_index = build_directed_edge_index(train_pos, num_nodes, add_self_loops=True).to(device)

    trials = DEFAULT_TRIALS[: max(1, args.num_trials)]
    if args.quick:
        trials = [DEFAULT_TRIALS[0], DEFAULT_TRIALS[1]]
    if args.single_trial_index > 0:
        idx = args.single_trial_index - 1
        if idx < 0 or idx >= len(DEFAULT_TRIALS):
            raise ValueError(f"single_trial_index must be in [1, {len(DEFAULT_TRIALS)}]")
        trials = [DEFAULT_TRIALS[idx]]

    trial_results = []
    for i, cfg in enumerate(trials, start=1):
        trial_out = train_one_trial(
            trial_id=i,
            cfg=cfg,
            x=x,
            train_graph_edge_index=train_graph_edge_index,
            train_pos=train_pos,
            val_pos=val_pos,
            val_neg=val_neg,
            all_positive_set=all_positive_set,
            neighbors=neighbors,
            device=device,
            epochs=args.epochs if not args.quick else min(args.epochs, 30),
            patience=args.patience if not args.quick else min(args.patience, 10),
            seed=args.seed,
            train_pos_sample_ratio=args.train_pos_sample_ratio,
        )
        trial_results.append(trial_out)

    best_trial = max(trial_results, key=lambda d: d["best_val_auc"])

    best_cfg = best_trial["config"]
    encoder = AdvancedGATEncoder(
        in_dim=x.size(1),
        hidden_dim=best_cfg.hidden_dim,
        out_dim=best_cfg.embed_dim,
        heads=best_cfg.heads,
        dropout=best_cfg.dropout,
    )
    decoder = EdgeMLPDecoder(
        dim=best_cfg.embed_dim,
        hidden_dim=best_cfg.decoder_hidden_dim,
        dropout=best_cfg.dropout,
    )
    best_model = LinkPredictionModel(encoder=encoder, decoder=decoder).to(device)
    best_model.load_state_dict(best_trial["best_state"])

    val_eval = evaluate_split(
        model=best_model,
        x=x,
        graph_edge_index=train_graph_edge_index,
        pos_edges=val_pos,
        neg_edges=val_neg,
        device=device,
    )
    test_eval = evaluate_split(
        model=best_model,
        x=x,
        graph_edge_index=train_graph_edge_index,
        pos_edges=test_pos,
        neg_edges=test_neg,
        device=device,
    )

    best_threshold, val_thr_metrics = find_best_threshold_by_f1(val_eval["y_true"], val_eval["y_prob"])
    test_metrics = binary_metrics(test_eval["y_true"], test_eval["y_prob"], best_threshold)

    target_auc = 0.875
    target_f1 = 0.850
    pass_auc = float(test_eval["auc"]) >= target_auc
    pass_f1 = float(test_metrics["f1"]) >= target_f1

    plot_best_trial_curve(best_trial["history"], str(Path(args.output_dir) / "best_trial_curves.png"))

    trials_table = []
    for r in trial_results:
        c = r["config"]
        trials_table.append(
            {
                "trial_id": int(r["trial_id"]),
                "hidden_dim": int(c.hidden_dim),
                "heads": int(c.heads),
                "dropout": float(c.dropout),
                "embed_dim": int(c.embed_dim),
                "decoder_hidden_dim": int(c.decoder_hidden_dim),
                "lr": float(c.lr),
                "weight_decay": float(c.weight_decay),
                "neg_ratio": float(c.neg_ratio),
                "hard_fraction": float(c.hard_fraction),
                "best_epoch": int(r["best_epoch"]),
                "best_val_auc": float(r["best_val_auc"]),
            }
        )

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
        },
        "search": {
            "num_trials": int(len(trial_results)),
            "quick_mode": bool(args.quick),
            "trials": trials_table,
        },
        "best_trial": {
            "trial_id": int(best_trial["trial_id"]),
            "best_epoch": int(best_trial["best_epoch"]),
            "best_val_auc": float(best_trial["best_val_auc"]),
            "config": {
                "hidden_dim": int(best_cfg.hidden_dim),
                "heads": int(best_cfg.heads),
                "dropout": float(best_cfg.dropout),
                "embed_dim": int(best_cfg.embed_dim),
                "decoder_hidden_dim": int(best_cfg.decoder_hidden_dim),
                "lr": float(best_cfg.lr),
                "weight_decay": float(best_cfg.weight_decay),
                "neg_ratio": float(best_cfg.neg_ratio),
                "hard_fraction": float(best_cfg.hard_fraction),
            },
        },
        "threshold_selected_on_val": float(best_threshold),
        "val_metrics_at_selected_threshold": val_thr_metrics,
        "test": {
            "auc": float(test_eval["auc"]),
            "f1": float(test_metrics["f1"]),
            "precision": float(test_metrics["precision"]),
            "recall": float(test_metrics["recall"]),
            "accuracy": float(test_metrics["accuracy"]),
        },
        "baseline_targets": {
            "auc_target": target_auc,
            "f1_target": target_f1,
            "pass_auc": pass_auc,
            "pass_f1": pass_f1,
            "pass_both": bool(pass_auc and pass_f1),
        },
    }

    save_json(results, str(Path(args.output_dir) / "q4_metrics.json"))
    torch.save(best_trial["best_state"], str(Path(args.output_dir) / "best_q4_model.pt"))

    print("Q4 completed")
    print(f"Best val AUC: {best_trial['best_val_auc']:.4f} (trial {best_trial['trial_id']})")
    print(f"Test AUC: {test_eval['auc']:.4f}, Test F1: {test_metrics['f1']:.4f}")
    print(
        f"Baseline pass -> AUC>=0.875: {pass_auc}, F1>=0.850: {pass_f1}, both: {bool(pass_auc and pass_f1)}"
    )
    print(f"Artifacts saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
