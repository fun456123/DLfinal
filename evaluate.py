from __future__ import annotations

import argparse
from collections import Counter
import json

import torch
from torch.utils.data import DataLoader, Subset

from src.config import parse_args_with_config
from src.data import DATASET_NAMES, PairedTransform, build_dataset, normalize_dataset_names
from src.engine import evaluate, evaluate_by_generator, load_model_weights
from src.branch_c import PatchForensicBranch
from src.fusion import FusionForensicDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Branch A + Branch C fusion checkpoint")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default="cifake",
        help=(
            "Dataset(s) to evaluate on. Use one of "
            f"{', '.join(DATASET_NAMES)}, or pass both names / 'both' to merge them."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "val"],
        default="test",
        help="Default split. With --dataset both --split test, Tiny-GenImage uses val.",
    )
    parser.add_argument(
        "--cifake-split",
        choices=["train", "test", "val"],
        default=None,
        help="Override --split for CIFAKE.",
    )
    parser.add_argument(
        "--tiny-genimage-split",
        choices=["train", "test", "val"],
        default=None,
        help="Override --split for Tiny-GenImage.",
    )
    parser.add_argument("--generators", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--semantic-size", type=int, default=224)
    parser.add_argument("--forensic-size", type=int, default=None)
    parser.add_argument("--branch-a-backbone", choices=["resnet18", "resnet34", "resnet50"], default="resnet18")
    parser.add_argument("--branch-a-feature-dim", type=int, default=128)
    parser.add_argument("--pretrained-branch-a", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-branch-a", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--branch-c-feature-dim", "--feature-dim", dest="branch_c_feature_dim", type=int, default=128)
    parser.add_argument("--freeze-branch-c", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-dropout", type=float, default=0.3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print evaluation progress every N batches. Use 0 to disable.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args_with_config(parser)
    args.dataset = normalize_dataset_names(args.dataset)
    if not args.checkpoint:
        parser.error("--checkpoint is required, either in config JSON or on the command line.")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    transform = PairedTransform(
        semantic_size=args.semantic_size,
        forensic_size=args.forensic_size,
        train=False,
        augment=False,
    )
    splits = _resolve_dataset_splits(args)
    dataset = build_dataset(args.dataset_root, args.dataset, splits, transform, generators=args.generators)
    full_dataset_counts = _count_dataset_records(dataset)
    if args.max_samples:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    dataset_counts = _count_dataset_records(dataset)
    missing_datasets = [name for name in args.dataset if full_dataset_counts.get(name, 0) == 0]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Datasets: {', '.join(args.dataset)}", flush=True)
    print(f"Splits: {splits}", flush=True)
    print(f"Dataset counts: {dict(dataset_counts)}", flush=True)
    if args.max_samples:
        print(f"Full split counts before --max-samples: {dict(full_dataset_counts)}", flush=True)
    if missing_datasets:
        print(
            "Warning: no records were found for "
            + ", ".join(f"{name} with split={splits[name]}" for name in missing_datasets)
            + ".",
            flush=True,
        )
    print(f"Evaluating {len(dataset)} images in {len(loader)} batches.", flush=True)
    print(f"Progress update: every {args.progress_every} batches", flush=True)
    print(f"Device: {device}", flush=True)
    branch_c = PatchForensicBranch(
        patch_size=args.patch_size,
        stride=args.stride,
        top_k=args.top_k,
        feature_dim=args.branch_c_feature_dim,
    )
    model = FusionForensicDetector(
        branch_c=branch_c,
        branch_a_backbone=args.branch_a_backbone,
        branch_a_feature_dim=args.branch_a_feature_dim,
        branch_c_feature_dim=args.branch_c_feature_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_dropout=args.fusion_dropout,
        pretrained_branch_a=args.pretrained_branch_a,
        freeze_branch_a=args.freeze_branch_a,
        freeze_branch_c=args.freeze_branch_c,
    ).to(device)
    checkpoint = load_model_weights(args.checkpoint, model, device)
    metrics = evaluate(model, loader, device, progress_every=args.progress_every)
    result = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "datasets": args.dataset,
        "splits": splits,
        "dataset_counts": dict(dataset_counts),
        "overall": metrics,
        "by_generator": evaluate_by_generator(model, loader, device, progress_every=args.progress_every),
    }
    print(json.dumps(result, indent=2))


def _count_dataset_records(dataset: object) -> Counter[str]:
    if isinstance(dataset, Subset):
        source = dataset.dataset
        if hasattr(source, "records"):
            return Counter(source.records[index].dataset for index in dataset.indices)
        return _count_dataset_records(source)
    if hasattr(dataset, "records"):
        return Counter(record.dataset for record in dataset.records)
    return Counter()


def _resolve_dataset_splits(args: argparse.Namespace) -> dict[str, str]:
    splits = {name: args.split for name in args.dataset}
    if len(args.dataset) > 1 and args.split == "test" and "tiny-genimage" in splits:
        splits["tiny-genimage"] = "val"
    if args.cifake_split is not None and "cifake" in splits:
        splits["cifake"] = args.cifake_split
    if args.tiny_genimage_split is not None and "tiny-genimage" in splits:
        splits["tiny-genimage"] = args.tiny_genimage_split
    return splits


if __name__ == "__main__":
    main()
