from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from src.branch_a import SemanticBranchA


class FusionForensicDetector(nn.Module):
    """
    Multi-view forensic detector.

    This model combines:
        Branch A: semantic / global feature branch
        Branch B: patch-level forensic feature branch

    Expected batch input:
        {
            "image_semantic": Tensor, shape (B, 3, Hs, Ws)
            "image_forensic": Tensor, shape (B, 3, Hf, Wf)
            "label": Tensor, shape (B,)
        }

    Expected Branch B interface:
        branch_b.extract_features(image_forensic) -> Tensor, shape (B, branch_b_feature_dim)

    Output:
        {
            "logits": Tensor, shape (B,)
            "feature_a": Tensor, shape (B, branch_a_feature_dim)
            "feature_b": Tensor, shape (B, branch_b_feature_dim)
            "features": Tensor, shape (B, branch_a_feature_dim + branch_b_feature_dim)
        }
    """

    def __init__(
        self,
        branch_b: nn.Module,
        branch_a_backbone: Literal["resnet18", "resnet34", "resnet50"] = "resnet18",
        branch_a_feature_dim: int = 128,
        branch_b_feature_dim: int = 128,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.3,
        pretrained_branch_a: bool = True,
        freeze_branch_a: bool = False,
        freeze_branch_b: bool = False,
    ) -> None:
        super().__init__()

        self.branch_a_feature_dim = branch_a_feature_dim
        self.branch_b_feature_dim = branch_b_feature_dim
        self.fusion_input_dim = branch_a_feature_dim + branch_b_feature_dim

        self.branch_a = SemanticBranchA(
            backbone=branch_a_backbone,
            feature_dim=branch_a_feature_dim,
            pretrained=pretrained_branch_a,
            dropout=fusion_dropout,
            freeze_backbone=freeze_branch_a,
        )

        self.branch_b = branch_b

        if freeze_branch_b:
            for param in self.branch_b.parameters():
                param.requires_grad = False

        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.BatchNorm1d(fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            batch:
                batch["image_semantic"]: semantic/global image view
                batch["image_forensic"]: forensic/local image view

        Returns:
            dict containing logits and intermediate features.
        """

        image_semantic = batch["image_semantic"]
        image_forensic = batch["image_forensic"]

        feature_a = self.branch_a(image_semantic)
        feature_b = self._extract_branch_b_features(image_forensic)

        self._validate_feature_shape(
            feature=feature_a,
            expected_dim=self.branch_a_feature_dim,
            name="Branch A feature",
        )
        self._validate_feature_shape(
            feature=feature_b,
            expected_dim=self.branch_b_feature_dim,
            name="Branch B feature",
        )

        fused_feature = torch.cat([feature_a, feature_b], dim=1)
        logits = self.fusion_classifier(fused_feature).squeeze(1)

        return {
            "logits": logits,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "features": fused_feature,
        }

    def _extract_branch_b_features(self, image_forensic: torch.Tensor) -> torch.Tensor:
        """
        Extract Branch B features.

        Preferred Branch B interface:
            branch_b.extract_features(image_forensic)

        Backup compatible interface:
            branch_b(image_forensic) returns a dict with key "features".
        """

        if hasattr(self.branch_b, "extract_features"):
            feature_b = self.branch_b.extract_features(image_forensic)
            return feature_b

        outputs = self.branch_b(image_forensic)

        if not isinstance(outputs, dict):
            raise TypeError(
                "Branch B must either implement extract_features(image_forensic), "
                "or forward(image_forensic) must return a dict containing key 'features'."
            )

        if "features" not in outputs:
            raise KeyError(
                "Branch B output dict must contain key 'features' when extract_features() is not implemented."
            )

        return outputs["features"]

    @staticmethod
    def _validate_feature_shape(
        feature: torch.Tensor,
        expected_dim: int,
        name: str,
    ) -> None:
        if feature.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D tensor with shape (batch_size, feature_dim), "
                f"but got shape {tuple(feature.shape)}."
            )

        if feature.shape[1] != expected_dim:
            raise ValueError(
                f"{name} dimension mismatch. "
                f"Expected feature_dim={expected_dim}, but got {feature.shape[1]}."
            )


class FusionClassifier(nn.Module):
    """
    Standalone fusion classifier.

    This class is optional. It can be used if features from Branch A and Branch B
    are already computed outside the model.

    Input:
        feature_a: Tensor, shape (B, branch_a_feature_dim)
        feature_b: Tensor, shape (B, branch_b_feature_dim)

    Output:
        logits: Tensor, shape (B,)
    """

    def __init__(
        self,
        branch_a_feature_dim: int = 128,
        branch_b_feature_dim: int = 128,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        fusion_input_dim = branch_a_feature_dim + branch_b_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.BatchNorm1d(fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )

    def forward(
        self,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
    ) -> torch.Tensor:
        fused_feature = torch.cat([feature_a, feature_b], dim=1)
        logits = self.classifier(fused_feature).squeeze(1)
        return logits