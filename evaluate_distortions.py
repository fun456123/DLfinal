from __future__ import annotations

import argparse
import csv
from functools import partial
from io import BytesIO
import json
from pathlib import Path
from typing import Callable

from PIL import Image, ImageFilter
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as TF

from src.branch_a import BranchAClassifier
from src.branch_b import PatchForensicBranch
from src.config import parse_args_with_config
from src.data import PairedTransform, build_dataset
from src.engine import evaluate, load_model_weights
from src.fusion import FusionForensicDetector


Distortion = Callable[[Image.Image], Image.Image]


class DistortedPairedTransform:
    def __init__(self, base_transform: PairedTransform, distortion: Distortion) -> None:
        self.base_transform = base_transform
        self.distortion = distortion

    def __call__(self, image: Image.Image) -> dict[str, torch.Tensor]:
        image = image.convert("RGB")
        image = self.distortion(image)
        return self.base_transform(image)


class BranchAForBatch(nn.Module):
    def __init__(self, model: BranchAClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model(batch["image_semantic"])


class BranchBForBatch(nn.Module):
    def __init__(self, model: PatchForensicBranch) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model(batch["image_forensic"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on distorted CIFAKE test images.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--dataset", default="cifake")
    parser.add_argument("--split", default="test", choices=["train", "test", "val"])
    parser.add_argument("--model-type", choices=["fusion", "branch-a", "branch-b"], default="fusion")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
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
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parse_args_with_config(parser)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    distortions = build_distortions()
    model = build_model(args).to(device)
    checkpoint = load_model_weights(args.checkpoint, model, device)

    results: list[dict[str, object]] = []
    for name, distortion in distortions.items():
        print(f"\n=== Distortion: {name} ===", flush=True)
        base_transform = PairedTransform(
            semantic_size=args.semantic_size,
            forensic_size=args.forensic_size,
            train=False,
            augment=False,
        )
        transform = DistortedPairedTransform(base_transform, distortion)
        dataset = build_dataset(args.dataset_root, args.dataset, args.split, transform, generators=None)
        if args.max_samples:
            dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"Evaluating {len(dataset)} images in {len(loader)} batches.", flush=True)
        metrics = evaluate(model, loader, device, progress_every=args.progress_every)
        row = {
            "distortion": name,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "dataset": args.dataset,
            "split": args.split,
            "num_samples": len(dataset),
            **metrics,
        }
        results.append(row)
        print(json.dumps(row, indent=2), flush=True)

    write_outputs(results, args)


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model_type == "branch-a":
        return BranchAForBatch(
            BranchAClassifier(
                backbone=args.branch_a_backbone,
                feature_dim=args.branch_a_feature_dim,
                pretrained=args.pretrained_branch_a,
                dropout=args.fusion_dropout,
                freeze_backbone=args.freeze_branch_a,
            )
        )

    branch_b = PatchForensicBranch(
        patch_size=args.patch_size,
        stride=args.stride,
        top_k=args.top_k,
        feature_dim=args.branch_b_feature_dim,
    )
    if args.model_type == "branch-b":
        return BranchBForBatch(branch_b)

    return FusionForensicDetector(
        branch_b=branch_b,
        branch_a_backbone=args.branch_a_backbone,
        branch_a_feature_dim=args.branch_a_feature_dim,
        branch_b_feature_dim=args.branch_b_feature_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_dropout=args.fusion_dropout,
        pretrained_branch_a=args.pretrained_branch_a,
        freeze_branch_a=args.freeze_branch_a,
        freeze_branch_b=args.freeze_branch_b,
    )


def build_distortions() -> dict[str, Distortion]:
    return {
        "clean": identity,
        "jpeg_q90": partial(jpeg_compress, quality=90),
        "jpeg_q70": partial(jpeg_compress, quality=70),
        "jpeg_q50": partial(jpeg_compress, quality=50),
        "jpeg_q30": partial(jpeg_compress, quality=30),
        "blur": partial(gaussian_blur, radius=1.0),
        "resize": partial(resize_down_up, scale=0.5),
        "noise": partial(add_gaussian_noise, std=0.05),
    }


def identity(image: Image.Image) -> Image.Image:
    return image.copy()


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def resize_down_up(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    down_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    down = image.resize(down_size, Image.Resampling.BICUBIC)
    return down.resize((width, height), Image.Resampling.BICUBIC)


def add_gaussian_noise(image: Image.Image, std: float) -> Image.Image:
    tensor = TF.to_tensor(image)
    noisy = (tensor + torch.randn_like(tensor) * std).clamp(0.0, 1.0)
    return TF.to_pil_image(noisy)


def write_outputs(results: list[dict[str, object]], args: argparse.Namespace) -> None:
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {output_json}", flush=True)

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(results[0].keys()) if results else []
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote CSV: {output_csv}", flush=True)


if __name__ == "__main__":
    main()
