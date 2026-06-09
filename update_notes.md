# Branch A + Branch B Fusion Model Update Notes

這份文件說明目前為了完成 `Branch A + Branch B` 主架構，我新增與修改了哪些檔案，以及 Branch B 需要提供的 feature interface。

目前設計重點是：

```text
image_semantic → Branch A → feature_a
image_forensic → Branch B → feature_b
concat(feature_a, feature_b) → fusion classifier → real/fake logits
```

也就是訓練時不是只訓練 fusion classifier，而是會讓 Branch A、Branch B、fusion classifier 一起 end-to-end training，除非之後在 config 裡設定 freeze。

---

## 1. 新增檔案

### 1.1 `src/branch_a.py`

新增 Branch A，也就是 semantic / global feature branch。

目前 Branch A 使用 torchvision 的 ResNet backbone，例如：

```python
resnet18
resnet34
resnet50
```

主要用途：

```text
batch["image_semantic"] → ResNet backbone → projection layer → feature_a
```

輸出格式：

```python
feature_a.shape == (batch_size, branch_a_feature_dim)
```

目前預設：

```python
branch_a_feature_dim = 128
```

Branch A 不直接負責最後分類，而是輸出 feature 給 fusion model 使用。

---

### 1.2 `src/fusion_model.py`

新增完整主架構 `FusionForensicDetector`。

它會同時使用：

```text
Branch A: semantic/global branch
Branch B: patch-level forensic branch
```

模型輸入是一整個 batch dictionary：

```python
{
    "image_semantic": Tensor,  # shape: (B, 3, Hs, Ws)
    "image_forensic": Tensor,  # shape: (B, 3, Hf, Wf)
    "label": Tensor            # shape: (B,)
}
```

模型輸出：

```python
{
    "logits": logits,          # shape: (B,)
    "feature_a": feature_a,    # shape: (B, branch_a_feature_dim)
    "feature_b": feature_b,    # shape: (B, branch_b_feature_dim)
    "features": fused_feature  # shape: (B, branch_a_feature_dim + branch_b_feature_dim)
}
```

其中 `logits` 是給 `BCEWithLogitsLoss()` 使用的 binary classification logit，所以 shape 必須是：

```python
(batch_size,)
```

不是：

```python
(batch_size, 2)
```

---

## 2. 修改檔案

### 2.1 `src/engine.py`

原本 `engine.py` 只支援 Branch B-only，會寫死使用：

```python
batch["image_forensic"]
```

現在改成支援完整 fusion model。

新的訓練與評估流程會直接把整個 batch 丟進 model：

```python
outputs = model(batch)
logits = outputs["logits"]
```

因此 model 必須支援：

```python
model(batch)
```

並且回傳 dict，其中一定要有：

```python
outputs["logits"]
```

目前新版 `engine.py` 不再支援舊寫法：

```python
model(batch["image_forensic"])
```

也就是如果 Branch B 還是舊版、沒有正確接到 fusion model，就會直接報錯。

---

### 2.2 `train.py`

原本 `train.py` 是訓練 Branch B-only。

現在改成訓練完整 fusion model：

```text
Branch A + Branch B → fusion classifier
```

也就是 `train.py` 會建立：

```python
branch_b = PatchForensicBranch(...)
model = FusionForensicDetector(branch_b=branch_b, ...)
```

訓練時 optimizer 使用：

```python
model.parameters()
```

所以預設情況下會 end-to-end 更新：

```text
Branch A parameters
Branch B parameters
Fusion classifier parameters
```

除非在 config 裡面設定：

```json
"freeze_branch_a": true
```

或：

```json
"freeze_branch_b": true
```

---

### 2.3 `evaluate.py`

原本 `evaluate.py` 只會建立 Branch B-only model 來載入 checkpoint。

現在改成建立完整 fusion model：

```python
branch_b = PatchForensicBranch(...)
model = FusionForensicDetector(branch_b=branch_b, ...)
```

因此 evaluation checkpoint 必須是 fusion model 訓練出來的 checkpoint。

---

## 3. 新增 config 檔案

### 3.1 `configs/cifake_fusion_a_b.json`

用於 CIFAKE 的 fusion model training。

建議內容：

```json
{
  "dataset_root": "dataset",
  "dataset": "cifake",

  "epochs": 10,
  "batch_size": 128,
  "lr": 0.0003,
  "weight_decay": 0.0001,
  "num_workers": 2,
  "val_fraction": 0.1,

  "semantic_size": 224,
  "forensic_size": null,

  "branch_a_backbone": "resnet18",
  "branch_a_feature_dim": 128,
  "pretrained_branch_a": true,
  "freeze_branch_a": false,

  "patch_size": 16,
  "stride": 8,
  "top_k": 4,
  "branch_b_feature_dim": 128,
  "freeze_branch_b": false,

  "fusion_hidden_dim": 256,
  "fusion_dropout": 0.3,

  "output_dir": "runs/cifake_fusion_a_b",
  "seed": 42,
  "device": "cuda"
}
```

---

### 3.2 `configs/tiny_sdv5_fusion_a_b.json`

用於 Tiny-GenImage，訓練 generator 為 `sdv5` 的 fusion model。

建議內容：

```json
{
  "dataset_root": "dataset",
  "dataset": "tiny-genimage",
  "generators": ["sdv5"],

  "epochs": 10,
  "batch_size": 64,
  "lr": 0.0003,
  "weight_decay": 0.0001,
  "num_workers": 2,
  "val_fraction": 0.1,

  "semantic_size": 224,
  "forensic_size": 224,

  "branch_a_backbone": "resnet18",
  "branch_a_feature_dim": 128,
  "pretrained_branch_a": true,
  "freeze_branch_a": false,

  "patch_size": 32,
  "stride": 16,
  "top_k": 8,
  "branch_b_feature_dim": 128,
  "freeze_branch_b": false,

  "fusion_hidden_dim": 256,
  "fusion_dropout": 0.3,

  "output_dir": "runs/tiny_sdv5_fusion_a_b",
  "seed": 42,
  "device": "cuda"
}
```

---

### 3.3 `configs/eval_cifake_fusion_a_b.json`

用於 evaluate CIFAKE fusion model checkpoint。

建議內容：

```json
{
  "checkpoint": "runs/cifake_fusion_a_b/best.pt",

  "dataset_root": "dataset",
  "dataset": "cifake",
  "split": "test",

  "batch_size": 128,
  "num_workers": 2,

  "semantic_size": 224,
  "forensic_size": null,

  "branch_a_backbone": "resnet18",
  "branch_a_feature_dim": 128,
  "pretrained_branch_a": true,
  "freeze_branch_a": false,

  "patch_size": 16,
  "stride": 8,
  "top_k": 4,
  "branch_b_feature_dim": 128,
  "freeze_branch_b": false,

  "fusion_hidden_dim": 256,
  "fusion_dropout": 0.3,

  "device": "cuda"
}
```

---

## 4. Branch B 必須提供的 interface

目前 fusion model 不再相容舊的 Branch B 寫法。

Branch B 一定要提供：

```python
extract_features(image_forensic)
```

也就是：

```python
feature_b = branch_b.extract_features(image_forensic)
```

### 4.1 輸入格式

```python
image_forensic.shape == (batch_size, 3, H, W)
```

其中：

- CIFAKE 預設 forensic image 可能是原始 32×32。
- Tiny-GenImage 預設 forensic image 會 resize 到 224×224。

實際大小由 config 控制：

```json
"forensic_size": null
```

或：

```json
"forensic_size": 224
```

---

### 4.2 輸出格式

Branch B 的 `extract_features()` 必須回傳一個 2D tensor：

```python
feature_b.shape == (batch_size, branch_b_feature_dim)
```

目前 config 預設：

```python
branch_b_feature_dim = 128
```

所以預期輸出是：

```python
feature_b.shape == (batch_size, 128)
```

---

### 4.3 最小可接受範例

Branch B 至少要長得像這樣：

```python
from __future__ import annotations

import torch
from torch import nn


class PatchForensicBranch(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        stride: int = 8,
        top_k: int = 4,
        feature_dim: int = 128,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.top_k = top_k
        self.feature_dim = feature_dim

        # TODO: implement patch selector, patch encoders, attention pooling, etc.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, feature_dim),
            nn.ReLU(inplace=True),
        )

        # Optional: useful for Branch B-only ablation.
        self.classifier = nn.Linear(feature_dim, 1)

    def extract_features(self, image_forensic: torch.Tensor) -> torch.Tensor:
        feature_b = self.feature_extractor(image_forensic)

        if feature_b.ndim != 2:
            raise ValueError(
                f"Branch B feature must be 2D, but got shape {tuple(feature_b.shape)}."
            )

        if feature_b.shape[1] != self.feature_dim:
            raise ValueError(
                f"Branch B feature_dim mismatch. Expected {self.feature_dim}, "
                f"but got {feature_b.shape[1]}."
            )

        return feature_b

    def forward(self, image_forensic: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_b = self.extract_features(image_forensic)
        logits = self.classifier(feature_b).squeeze(1)

        return {
            "logits": logits,
            "features": feature_b,
        }
```

注意：上面只是最小格式範例，不是最終 Branch B 架構。真正的 Branch B 還是應該使用 patch-level forensic method，例如：

```text
image_forensic
 → unfold into patches
 → FFT frequency energy score
 → select top-K high-frequency patches
 → select bottom-K low-frequency patches
 → high/low patch encoders
 → attention pooling
 → feature_b
```

---

## 5. 訓練與測試方式

### 5.1 CIFAKE smoke test

在正式訓練前，建議先跑小資料測試：

```bash
python train.py \
  --config configs/cifake_fusion_a_b.json \
  --epochs 1 \
  --batch-size 16 \
  --num-workers 0 \
  --max-train-samples 64 \
  --max-val-samples 32 \
  --output-dir runs/smoke_fusion_a_b
```

如果 smoke test 成功，再正式跑：

```bash
python train.py --config configs/cifake_fusion_a_b.json
```

---

### 5.2 CIFAKE evaluation

```bash
python evaluate.py --config configs/eval_cifake_fusion_a_b.json
```

---

### 5.3 Tiny-GenImage training

```bash
python train.py --config configs/tiny_sdv5_fusion_a_b.json
```

---

## 6. 重要注意事項

### 6.1 `logits` shape

整個專案目前使用 binary classification 的設計，loss 是：

```python
nn.BCEWithLogitsLoss()
```

因此模型最後輸出必須是：

```python
logits.shape == (batch_size,)
```

不要輸出：

```python
(batch_size, 2)
```

---

### 6.2 label format

目前 label 是：

```python
0 = REAL / nature
1 = FAKE / ai
```

所以 logits 經過 sigmoid 後代表 fake probability：

```python
prob_fake = torch.sigmoid(logits)
```

---

### 6.3 Branch B feature dimension 必須對齊 config

如果 config 寫：

```json
"branch_b_feature_dim": 128
```

那 Branch B 的 `extract_features()` 就一定要回傳：

```python
(batch_size, 128)
```

否則 fusion model 會直接報錯。

---

### 6.4 現在不再支援舊 Branch B

新版 `engine.py`、`train.py`、`evaluate.py` 都是以完整 fusion model 為主。

也就是說：

```python
model(batch)
```

是必要接口。

不再支援：

```python
model(batch["image_forensic"])
```

如果 Branch B 沒有提供正確的 `extract_features()`，整個訓練會直接報錯，這是故意設計的，目的是避免 silent bug 或 feature 接錯。
