from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FrequencyPatchSelector(nn.Module):
    """Select high-frequency and low-frequency patches using FFT energy."""

    def __init__(self, patch_size: int = 16, stride: int = 8, top_k: int = 4, high_radius_ratio: float = 0.35) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.top_k = top_k
        self.high_radius_ratio = high_radius_ratio

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = F.unfold(images, kernel_size=self.patch_size, stride=self.stride)
        batch_size, channels_times_area, num_patches = patches.shape
        channels = images.shape[1]
        patches = patches.transpose(1, 2).reshape(
            batch_size,
            num_patches,
            channels,
            self.patch_size,
            self.patch_size,
        )

        scores = self._frequency_scores(patches)
        k = min(self.top_k, num_patches)
        high_idx = scores.topk(k=k, dim=1, largest=True).indices
        low_idx = scores.topk(k=k, dim=1, largest=False).indices
        high_patches = self._gather_patches(patches, high_idx)
        low_patches = self._gather_patches(patches, low_idx)
        return high_patches, low_patches, scores

    def _frequency_scores(self, patches: torch.Tensor) -> torch.Tensor:
        gray = patches.mean(dim=2)
        spectrum = torch.fft.fftshift(torch.fft.fft2(gray, norm="ortho"), dim=(-2, -1)).abs().pow(2)
        size = spectrum.shape[-1]
        coords = torch.arange(size, device=patches.device) - (size - 1) / 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        radius = torch.sqrt(xx.pow(2) + yy.pow(2))
        high_mask = radius >= radius.max() * self.high_radius_ratio
        high_energy = spectrum[..., high_mask].sum(dim=-1)
        total_energy = spectrum.flatten(-2).sum(dim=-1).clamp_min(1e-8)
        return high_energy / total_energy

    @staticmethod
    def _gather_patches(patches: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch_size, _, channels, height, width = patches.shape
        gather_index = indices[:, :, None, None, None].expand(-1, -1, channels, height, width)
        return patches.gather(dim=1, index=gather_index)


class PatchEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, channels, height, width = patches.shape
        features = self.net(patches.reshape(batch_size * num_patches, channels, height, width))
        return features.reshape(batch_size, num_patches, -1)


class AttentionPool(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(features), dim=1)
        return (features * weights).sum(dim=1)


class PatchForensicBranch(nn.Module):
    def __init__(self, patch_size: int = 16, stride: int = 8, top_k: int = 4, feature_dim: int = 128) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.top_k = top_k
        self.feature_dim = feature_dim

        self.selector = FrequencyPatchSelector(patch_size=patch_size, stride=stride, top_k=top_k)
        self.high_encoder = PatchEncoder(feature_dim=feature_dim)
        self.low_encoder = PatchEncoder(feature_dim=feature_dim)
        self.high_pool = AttentionPool(feature_dim)
        self.low_pool = AttentionPool(feature_dim)
        fusion_dim = feature_dim * 4
        self.feature_projection = nn.Sequential(
            nn.Linear(fusion_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        feature_b, _ = self._extract_features_with_aux(images)
        return feature_b

    def _extract_features_with_aux(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        high_patches, low_patches, scores = self.selector(images)
        high_feat = self.high_pool(self.high_encoder(high_patches))
        low_feat = self.low_pool(self.low_encoder(low_patches))
        forensic_fusion = torch.cat(
            [high_feat, low_feat, torch.abs(high_feat - low_feat), high_feat * low_feat],
            dim=1,
        )
        feature_b = self.feature_projection(forensic_fusion)

        if feature_b.ndim != 2:
            raise ValueError(
                f"Branch B feature must be 2D, but got shape {tuple(feature_b.shape)}."
            )

        if feature_b.shape[1] != self.feature_dim:
            raise ValueError(
                f"Branch B feature_dim mismatch. Expected {self.feature_dim}, "
                f"but got {feature_b.shape[1]}."
            )

        aux = {
            "frequency_scores": scores,
            "high_feature": high_feat,
            "low_feature": low_feat,
            "raw_features": forensic_fusion,
        }
        return feature_b, aux

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_b, aux = self._extract_features_with_aux(images)
        logits = self.classifier(feature_b).squeeze(1)
        return {
            "logits": logits,
            "features": feature_b,
            **aux,
        }
