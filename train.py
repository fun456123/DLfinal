from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.config import parse_args_with_config, save_resolved_config
from src.data import DATASET_NAMES, PairedTransform, build_dataset, normalize_dataset_names, split_train_val
from src.engine import evaluate, save_checkpoint, train_one_epoch
from src.branch_b import PatchForensicBranch
from src.fusion import FusionForensicDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Branch A + Branch B fusion detector")
    parser.add_argument("--dataset-root", default="dataset", help="Root directory containing cifake/ and tiny-genimage/.")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default="cifake",
        help=(
            "Dataset(s) to train on. Use one of "
            f"{', '.join(DATASET_NAMES)}, or pass both names / 'both' to merge them."
        ),
    )
    parser.add_argument("--generators", nargs="*", default=None, help="Tiny-GenImage generator names to include.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--semantic-size", type=int, default=224)
    parser.add_argument("--forensic-size", type=int, default=None)
    parser.add_argument("--branch-a-backbone", choices=["resnet18", "resnet34", "resnet50"], default="resnet18")
    parser.add_argument("--branch-a-feature-dim", type=int, default=128)
    parser.add_argument("--pretrained-branch-a", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-branch-a", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--branch-b-feature-dim", "--feature-dim", dest="branch_b_feature_dim", type=int, default=128)
    parser.add_argument("--freeze-branch-b", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-dropout", type=float, default=0.3)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print train/validation progress every N batches. Use 0 to disable.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Useful for quick smoke tests.")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="runs/fusion_a_b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parse_args_with_config(parser)


def main() -> None:
    args = parse_args()
    args.dataset = normalize_dataset_names(args.dataset)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_transform = PairedTransform(
        semantic_size=args.semantic_size,
        forensic_size=args.forensic_size,
        train=True,
        augment=args.augment,
    )
    val_transform = PairedTransform(
        semantic_size=args.semantic_size,
        forensic_size=args.forensic_size,
        train=False,
        augment=False,
    )

    train_full = build_dataset(args.dataset_root, args.dataset, "train", train_transform, generators=args.generators)
    val_full_for_split = build_dataset(args.dataset_root, args.dataset, "train", val_transform, generators=args.generators)
    train_subset, val_subset = split_train_val(train_full, args.val_fraction, args.seed)
    _, val_indices = train_subset.indices, val_subset.indices
    val_dataset = Subset(val_full_for_split, val_indices)

    if args.max_train_samples:
        train_subset = Subset(train_subset, range(min(args.max_train_samples, len(train_subset))))
    if args.max_val_samples:
        val_dataset = Subset(val_dataset, range(min(args.max_val_samples, len(val_dataset))))

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    branch_b = PatchForensicBranch(
        patch_size=args.patch_size,
        stride=args.stride,
        top_k=args.top_k,
        feature_dim=args.branch_b_feature_dim,
    )
    model = FusionForensicDetector(
        branch_b=branch_b,
        branch_a_backbone=args.branch_a_backbone,
        branch_a_feature_dim=args.branch_a_feature_dim,
        branch_b_feature_dim=args.branch_b_feature_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_dropout=args.fusion_dropout,
        pretrained_branch_a=args.pretrained_branch_a,
        freeze_branch_a=args.freeze_branch_a,
        freeze_branch_b=args.freeze_branch_b,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(output_dir / "config.resolved.json", args)
    best_auroc = -1.0
    history: list[dict[str, object]] = []

    print(f"Datasets: {', '.join(args.dataset)}", flush=True)
    print(f"Augmentation: {'enabled' if args.augment else 'disabled'}", flush=True)
    print(f"Training on {len(train_subset)} images, validating on {len(val_dataset)} images.", flush=True)
    print(f"Train batches: {len(train_loader)}, validation batches: {len(val_loader)}", flush=True)
    print(f"Progress update: every {args.progress_every} batches", flush=True)
    print(f"Device: {device}", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=args.epochs,
            progress_every=args.progress_every,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            epoch=epoch,
            total_epochs=args.epochs,
            progress_every=args.progress_every,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)

        val_score = val_metrics["auroc"]
        if val_score != val_score:
            val_score = val_metrics["balanced_accuracy"]
        if val_score > best_auroc:
            best_auroc = val_score
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, val_metrics)

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_checkpoint(output_dir / "last.pt", model, optimizer, args.epochs, history[-1]["val"])


if __name__ == "__main__":
    main()
