# AI 生成影像偵測：Branch A + Branch Ｂ 雙分支融合模型

報告裡的branch B在這裡全部改稱branch c

本專案是一個用於 **AI-generated image detection** 的深度學習分類系統，目標是判斷輸入影像是真實影像還是 AI 生成影像。專案核心模型採用雙分支設計：

- **Branch A：Semantic / Global Feature Branch**  
  使用 ResNet 擷取整張影像的語意與全域視覺特徵。

- **Branch Ｃ：Patch-Level Forensic Branch**  
  使用 FFT 頻域能量挑選可疑 patch，擷取局部鑑識特徵。

- **Fusion Classifier：Feature-Level Fusion**  
  將 Branch A 與 Branch Ｃ 的特徵串接後，透過 MLP 輸出 real/fake 二元分類結果。

整體概念是：AI 生成影像不只可能在語意層面與真實影像不同，也可能在局部紋理、頻率分布、壓縮痕跡與細節一致性上留下痕跡。因此本專案同時觀察「整張圖的高階語意」與「局部 patch 的頻域鑑識訊號」。

## 專案目標

本專案主要完成以下功能：

- 支援 CIFAKE 與 Tiny-GenImage 兩種資料集格式。
- 支援單資料集訓練，也支援合併多資料集訓練。
- 提供 Branch A + Branch B 的融合式 AI 影像偵測模型。
- 使用 FFT high-frequency energy 進行 patch 選擇。
- 支援資料增強、訓練進度輸出、checkpoint 儲存。
- 提供整體評估、依 generator 分組評估，以及失真魯棒性評估。
- 輸出 accuracy、balanced accuracy、precision、recall、F1、AUROC 等指標。

## 專案結構

```text
DLfinal/
  configs/
    cifake_branch_c.json
    cifake_fusion_a_c.json
    tiny_sdv5_branch_c.json
    tiny_sdv5_fusion_a_c.json
    eval_cifake.json
    eval_cifake_fusion_a_c.json
    eval_tiny_sdv5_fusion_a_c.json
  dataset/
    cifake/
    tiny-genimage/
  runs/
    ...
  src/
    branch_a.py
    branch_c.py
    config.py
    data.py
    engine.py
    fusion.py
    metrics.py
  train.py
  evaluate.py
  evaluate_distortions.py
  requirements.txt
  README.md
```

主要檔案說明：

| 檔案 | 功能 |
| --- | --- |
| `train.py` | 訓練入口，負責讀取資料、建立模型、訓練、驗證與儲存 checkpoint。 |
| `evaluate.py` | 一般評估入口，載入 checkpoint 後在指定資料集 split 上計算指標。 |
| `evaluate_distortions.py` | 失真魯棒性評估，測試 JPEG、blur、resize、noise 等干擾。 |
| `src/data.py` | 建立資料列表、資料集類別、影像轉換與 train/val 切分。 |
| `src/branch_a.py` | Branch A 語意分支，使用 torchvision ResNet 作為 backbone。 |
| `src/branch_c.py` | Branch C patch 鑑識分支，包含 FFT patch selector、patch encoder、attention pooling。 |
| `src/fusion.py` | Branch A + Branch C 融合模型。 |
| `src/engine.py` | 訓練、驗證、checkpoint 儲存與載入。 |
| `src/metrics.py` | 二元分類指標計算。 |
| `src/config.py` | JSON config 載入與 resolved config 儲存。 |

## 環境需求

本專案使用 Python 與 PyTorch。`requirements.txt` 目前包含：

```text
torch
torchvision
pillow
```

安裝方式：

```bash
pip install -r requirements.txt
```

若要使用 GPU，請確認安裝的 PyTorch 版本與 CUDA 版本相容。

## 資料集格式

### CIFAKE

預期資料夾結構如下：

```text
dataset/
  cifake/
    train/
      REAL/
      FAKE/
    test/
      REAL/
      FAKE/
```

標籤定義：

| 資料夾 | Label | 意義 |
| --- | ---: | --- |
| `REAL` | `0` | 真實影像 |
| `FAKE` | `1` | AI 生成影像 |

在程式中，CIFAKE 的 generator 名稱會記為 `stable_diffusion_1_4`。

### Tiny-GenImage

預期資料夾結構如下：

```text
dataset/
  tiny-genimage/
    imagenet_ai_0419_biggan/
      train/
        nature/
        ai/
      val/
        nature/
        ai/
    imagenet_ai_0424_sdv5/
      train/
        nature/
        ai/
      val/
        nature/
        ai/
```

標籤定義：

| 資料夾 | Label | 意義 |
| --- | ---: | --- |
| `nature` | `0` | 真實影像 |
| `ai` | `1` | AI 生成影像 |

Tiny-GenImage 可使用 `--generators` 選擇特定生成器，例如：

```bash
python train.py --dataset tiny-genimage --generators sdv5
```

程式會將較長的資料夾名稱轉成短名稱，例如 `imagenet_ai_0424_sdv5` 會被簡化為 `sdv5`。

## 整體流程

整個訓練與推論流程如下：

```text
Input image
   |
   v
PairedTransform
   |
   +--> image_semantic: resize to semantic_size, ImageNet normalization
   |
   +--> image_forensic: optional resize to forensic_size, tensor only
   |
   v
Model
   |
   +--> Branch A: ResNet semantic feature
   |
   +--> Branch C: FFT patch selector + patch encoder + attention pooling
   |
   v
Feature concatenation
   |
   v
Fusion MLP classifier
   |
   v
Logit
   |
   v
Sigmoid probability
   |
   v
Real / Fake prediction
```

### 1. 資料掃描

`src/data.py` 會根據 `--dataset-root`、`--dataset`、`--split` 與 `--generators` 掃描影像檔案。支援的副檔名包含：

```text
.jpg, .jpeg, .png, .bmp, .webp
```

每張圖片會被包成 `ImageRecord`：

```python
ImageRecord(
    path=...,
    label=0 or 1,
    dataset="cifake" or "tiny-genimage",
    split="train" / "test" / "val",
    generator=...
)
```

### 2. 雙視角轉換

本專案使用 `PairedTransform` 從同一張 PIL image 產生兩種 view：

| View | 用途 | 前處理 |
| --- | --- | --- |
| `image_semantic` | 給 Branch A | RGB、resize 到 `semantic_size`、轉 tensor、ImageNet normalize |
| `image_forensic` | 給 Branch C | RGB、可選 resize 到 `forensic_size`、轉 tensor |

Branch A 使用 ImageNet normalization，是因為 ResNet backbone 預設使用 ImageNet 預訓練權重。Branch C 則保留較直接的像素 tensor，避免 normalization 改變局部鑑識訊號。

### 3. Train / Validation 切分

訓練時，程式會從指定資料集的 `train` split 中切出 validation set：

```text
train split
   |
   +--> training subset
   |
   +--> validation subset
```

切分比例由 `--val-fraction` 控制，預設為 `0.1`。切分使用 `--seed` 固定隨機性，預設 seed 為 `42`。

### 4. 資料增強

訓練時可用 `--augment` 啟用資料增強。增強只會套用在 training subset，不會套用在 validation 或 evaluation。

目前支援：

| 增強 | 機率 | 參數 |
| --- | ---: | --- |
| Random horizontal flip | `0.5` | 左右翻轉 |
| Gaussian blur | `0.25` | `kernel_size=3`, `sigma=(0.1, 0.8)` |

啟用方式：

```bash
python train.py --dataset cifake --augment
```

若 config 裡啟用了 augmentation，也可以用以下方式關閉：

```bash
python train.py --config configs/cifake_fusion_a_c.json --no-augment
```

## 網路設計

### Branch A：Semantic / Global Feature Branch

Branch A 實作於 `src/branch_a.py`，主要負責擷取整張影像的語意特徵。

架構：

```text
image_semantic
   |
   v
ResNet backbone
   |
   v
Remove final fc, output backbone feature
   |
   v
Linear(backbone_out_dim -> feature_dim)
   |
   v
BatchNorm1d
   |
   v
ReLU
   |
   v
Dropout
   |
   v
feature_a
```

支援的 backbone：

- `resnet18`
- `resnet34`
- `resnet50`

重要參數：

| 參數 | 預設值 | 說明 |
| --- | ---: | --- |
| `branch_a_backbone` | `resnet18` | ResNet backbone 類型 |
| `branch_a_feature_dim` | `128` | Branch A 輸出特徵維度 |
| `pretrained_branch_a` | `true` | 是否使用 ImageNet 預訓練權重 |
| `freeze_branch_a` | `false` | 是否凍結 ResNet backbone |
| `fusion_dropout` | `0.3` | Branch A projection 與 fusion classifier 使用的 dropout |

Branch A 的優點是能理解影像整體內容，例如物體形狀、場景語意、構圖與高階視覺特徵。不過只看整張圖可能忽略 AI 生成影像常出現在局部紋理與頻率分布上的細節破綻，因此本專案加入 Branch C。

### Branch C：Patch-Level Forensic Branch

Branch C 實作於 `src/branch_c.py`，是本專案的局部鑑識分支。它不直接處理整張圖，而是先切 patch，再根據 FFT 頻率能量挑選最有代表性的 patch。

Branch C 流程：

```text
image_forensic
   |
   v
F.unfold into patches
   |
   v
FFT frequency scoring
   |
   +--> select top-k high-frequency patches
   |
   +--> select top-k low-frequency patches
   |
   v
High patch encoder / Low patch encoder
   |
   v
Attention pooling
   |
   v
high_feat, low_feat
   |
   v
concat(high_feat, low_feat, abs(high-low), high*low)
   |
   v
Linear + LayerNorm + ReLU
   |
   v
feature_c
```

#### FrequencyPatchSelector

`FrequencyPatchSelector` 使用 `torch.nn.functional.unfold` 將影像切成重疊 patch：

```text
patch_size = 16 or 32
stride = 8 or 16
```

接著對每個 patch 做 FFT：

```python
spectrum = fftshift(fft2(gray_patch)).abs().pow(2)
```

每個 patch 會被轉為灰階後計算頻譜能量。程式以半徑遮罩區分高頻區域，並計算：

```text
score = high_frequency_energy / total_frequency_energy
```

分數越高代表該 patch 的高頻比例越高，可能包含邊緣、紋理、雜訊或生成痕跡；分數越低則代表該 patch 較平滑或低頻成分較多。

Branch C 會同時選：

- `top_k` 個最高頻 patch
- `top_k` 個最低頻 patch

這樣設計的原因是高頻 patch 可以捕捉局部紋理與 artifacts，低頻 patch 則可提供平滑區域、背景與整體色塊分布的對照訊息。

#### PatchEncoder

高頻 patch 與低頻 patch 分別送入各自的 CNN encoder：

```text
Conv2d(3 -> 32, 3x3)
BatchNorm2d
ReLU
Conv2d(32 -> 64, 3x3)
BatchNorm2d
ReLU
MaxPool2d
Conv2d(64 -> 128, 3x3)
BatchNorm2d
ReLU
AdaptiveAvgPool2d(1)
Flatten
Linear(128 -> feature_dim)
ReLU
```

每個 patch 會被編碼成 `feature_dim` 維特徵，預設為 `128`。

#### AttentionPool

因為每張圖會選出多個 patch，Branch C 使用 attention pooling 將多個 patch feature 聚合為單一向量。

Attention score 計算方式：

```text
Linear(feature_dim -> feature_dim / 2)
Tanh
Linear(feature_dim / 2 -> 1)
Softmax over patches
Weighted sum
```

最後得到：

```text
high_feat: high-frequency summary
low_feat: low-frequency summary
```

#### High / Low Feature Fusion

Branch C 不只單純串接 high 與 low features，還加入兩種交互特徵：

```text
forensic_fusion = concat(
    high_feat,
    low_feat,
    abs(high_feat - low_feat),
    high_feat * low_feat
)
```

其中：

- `high_feat`：高頻 patch 的鑑識特徵。
- `low_feat`：低頻 patch 的鑑識特徵。
- `abs(high_feat - low_feat)`：高低頻差異。
- `high_feat * low_feat`：高低頻交互關係。

接著經過：

```text
Linear(feature_dim * 4 -> feature_dim)
LayerNorm
ReLU
```

輸出 `feature_c`。

Branch C 也有獨立 classifier，可用於單分支實驗：

```text
Linear(feature_dim -> 256)
ReLU
Dropout(0.3)
Linear(256 -> 1)
```

### FusionForensicDetector：雙分支融合模型

融合模型實作於 `src/fusion.py`。

輸入 batch 格式：

```python
{
    "image_semantic": Tensor,  # shape: (B, 3, semantic_size, semantic_size)
    "image_forensic": Tensor,  # shape: (B, 3, Hf, Wf)
    "label": Tensor            # shape: (B,)
}
```

模型流程：

```text
image_semantic --> Branch A --> feature_a
image_forensic --> Branch C --> feature_c

feature_a + feature_c
   |
   v
concat along feature dimension
   |
   v
Fusion classifier
   |
   v
logit
```

Fusion classifier 架構：

```text
Linear(branch_a_feature_dim + branch_c_feature_dim -> fusion_hidden_dim)
BatchNorm1d
ReLU
Dropout
Linear(fusion_hidden_dim -> fusion_hidden_dim / 2)
BatchNorm1d
ReLU
Dropout
Linear(fusion_hidden_dim / 2 -> 1)
```

預設維度：

```text
branch_a_feature_dim = 128
branch_c_feature_dim = 128
fusion_input_dim = 256
fusion_hidden_dim = 256
```

輸出：

```python
{
    "logits": logits,
    "feature_a": feature_a,
    "feature_c": feature_c,
    "features": fused_feature
}
```

訓練時使用 `BCEWithLogitsLoss`，因此模型輸出的是 logit，不是 sigmoid 後的 probability。評估時計算 metrics 前會對 logit 套用 sigmoid，並以 `0.5` 作為分類 threshold。

## 訓練方法

查看所有參數：

```bash
python train.py --help
```

基本訓練指令：

```bash
python train.py --config configs/cifake_fusion_a_c.json
```

等價於使用 CIFAKE 訓練 Branch A + Branch C 融合模型。

### CIFAKE 訓練範例

```bash
python train.py \
  --dataset cifake \
  --epochs 10 \
  --batch-size 128 \
  --semantic-size 224 \
  --patch-size 16 \
  --stride 8 \
  --top-k 4 \
  --output-dir runs/cifake_fusion_a_c
```

### CIFAKE 加資料增強

```bash
python train.py \
  --dataset cifake \
  --augment \
  --epochs 10 \
  --batch-size 128 \
  --patch-size 16 \
  --stride 8 \
  --top-k 4 \
  --output-dir runs/cifake_aug_fusion_a_c
```

### Tiny-GenImage 指定 generator

```bash
python train.py \
  --dataset tiny-genimage \
  --generators sdv5 \
  --augment \
  --epochs 10 \
  --batch-size 64 \
  --forensic-size 224 \
  --patch-size 32 \
  --stride 16 \
  --top-k 8 \
  --output-dir runs/tiny_sdv5_fusion_a_c
```

### 合併 CIFAKE 與 Tiny-GenImage

```bash
python train.py \
  --dataset both \
  --augment \
  --epochs 10 \
  --batch-size 64 \
  --forensic-size 224 \
  --patch-size 32 \
  --stride 16 \
  --top-k 8 \
  --progress-every 10 \
  --output-dir runs/cifake_tiny_aug_fusion_a_c
```

也可以明確寫成：

```bash
python train.py --dataset cifake tiny-genimage
```

### 訓練輸出

每次訓練會在 `--output-dir` 下輸出：

| 檔案 | 說明 |
| --- | --- |
| `best.pt` | validation AUROC 最佳的 checkpoint。若 AUROC 無法計算，改用 balanced accuracy。 |
| `last.pt` | 最後一個 epoch 的 checkpoint。 |
| `history.json` | 每個 epoch 的 train / validation metrics。 |
| `config.resolved.json` | 實際使用的完整設定。 |

checkpoint 內容包含：

```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "metrics": metrics
}
```

## 評估方法

查看所有參數：

```bash
python evaluate.py --help
```

### CIFAKE test 評估

```bash
python evaluate.py \
  --checkpoint runs/cifake_fusion_a_c/best.pt \
  --dataset cifake \
  --split test
```

### Tiny-GenImage val 評估

```bash
python evaluate.py \
  --checkpoint runs/tiny_sdv5_fusion_a_c/best.pt \
  --dataset tiny-genimage \
  --split val \
  --generators sdv5 \
  --forensic-size 224 \
  --patch-size 32 \
  --stride 16 \
  --top-k 8
```

### 合併資料集評估

```bash
python evaluate.py \
  --checkpoint runs/cifake_tiny_aug_fusion_a_c/best.pt \
  --dataset both \
  --split test \
  --forensic-size 224 \
  --patch-size 32 \
  --stride 16 \
  --top-k 8 \
  --progress-every 10
```

評估輸出為 JSON，包含：

- checkpoint epoch
- dataset 與 split
- 每個 dataset 的 sample 數量
- overall metrics
- by-generator metrics

## 失真魯棒性評估

`evaluate_distortions.py` 用於測試模型在常見影像干擾下的穩定性。這些失真會在記憶體中套用，不會修改 `dataset/` 中的原始圖片。

支援的失真：

| 名稱 | 說明 |
| --- | --- |
| `clean` | 原始影像 |
| `jpeg_q90` | JPEG quality 90 |
| `jpeg_q70` | JPEG quality 70 |
| `jpeg_q50` | JPEG quality 50 |
| `jpeg_q30` | JPEG quality 30 |
| `blur` | Gaussian blur, radius 1.0 |
| `resize` | 先縮小到 0.5 倍再放回原大小 |
| `noise` | Gaussian noise, std 0.05 |

完整評估範例：

```bash
python evaluate_distortions.py \
  --checkpoint runs/cifake_tiny_aug_fusion_a_c/best.pt \
  --model-type fusion \
  --dataset cifake \
  --split test \
  --batch-size 64 \
  --forensic-size 224 \
  --patch-size 32 \
  --stride 16 \
  --top-k 8 \
  --progress-every 10 \
  --output-json runs/cifake_tiny_aug_fusion_a_c/distortion_results.json \
  --output-csv runs/cifake_tiny_aug_fusion_a_c/distortion_results.csv
```

快速 smoke test：

```bash
python evaluate_distortions.py \
  --checkpoint runs/cifake_tiny_aug_fusion_a_c/best.pt \
  --model-type fusion \
  --dataset cifake \
  --split test \
  --max-samples 128 \
  --batch-size 64 \
  --forensic-size 224 \
  --patch-size 32 \
  --stride 16 \
  --top-k 8
```

Windows 環境下，`evaluate_distortions.py` 預設 `--num-workers 0`，避免 multiprocessing pickling 問題。如果本機環境支援，可以手動提高：

```bash
python evaluate_distortions.py --num-workers 2 ...
```

## Config 使用方式

所有 CLI 都支援 `--config` 讀取 JSON 設定檔。設定檔中的 key 可使用底線或 CLI 對應名稱，程式會將 `-` 正規化為 `_`。

範例：

```bash
python train.py --config configs/cifake_fusion_a_c.json
```

也可以用 command line 覆蓋 config 內的設定：

```bash
python train.py \
  --config configs/cifake_fusion_a_c.json \
  --epochs 20 \
  --batch-size 64 \
  --output-dir runs/cifake_fusion_a_c_e20
```

## 重要超參數建議

### CIFAKE

建議設定：

| 參數 | 建議值 |
| --- | ---: |
| `semantic_size` | `224` |
| `forensic_size` | `null` |
| `patch_size` | `16` |
| `stride` | `8` |
| `top_k` | `4` |
| `batch_size` | `128` |
| `lr` | `3e-4` |
| `weight_decay` | `1e-4` |

原因是 CIFAKE 影像尺寸較小，直接保留 forensic view 原尺寸可避免過度 resize 改變局部頻率特徵。

### Tiny-GenImage

建議設定：

| 參數 | 建議值 |
| --- | ---: |
| `semantic_size` | `224` |
| `forensic_size` | `224` |
| `patch_size` | `32` |
| `stride` | `16` |
| `top_k` | `8` |
| `batch_size` | `64` |
| `lr` | `3e-4` |
| `weight_decay` | `1e-4` |

Tiny-GenImage 可能包含較大或不同尺寸影像，因此 forensic view 固定到 `224x224` 可以讓 patch 數量與模型輸入更穩定。

## 評估指標

本專案在 `src/metrics.py` 中實作以下二元分類指標：

| 指標 | 說明 |
| --- | --- |
| `accuracy` | 整體分類正確率。 |
| `balanced_accuracy` | real 與 fake recall 的平均，較能處理類別不平衡。 |
| `precision` | 預測為 fake 的樣本中，有多少是真的 fake。 |
| `recall` | 所有 fake 樣本中，有多少被成功抓出。 |
| `f1` | precision 與 recall 的 harmonic mean。 |
| `auroc` | 以 sigmoid probability 排序後計算 ROC AUC。 |
| `loss` | BCEWithLogitsLoss 平均值。 |

分類門檻固定為：

```text
sigmoid(logit) >= 0.5 -> fake
sigmoid(logit) < 0.5  -> real
```

## 目前實驗結果摘要

以下結果來自專案中已存在的 `runs/` 與 `eval_cifake_result.json`。

### CIFAKE Fusion A+C

`runs/cifake_fusion_a_c/history.json` 顯示，CIFAKE fusion 模型在 validation set 上表現穩定，最佳 validation AUROC 出現在 epoch 8：

| Epoch | Val Accuracy | Val F1 | Val AUROC | Val Loss |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.9757 | 0.9762 | 0.9977 | 0.0711 |
| 9 | 0.9781 | 0.9783 | 0.9976 | 0.0643 |
| 10 | 0.9773 | 0.9774 | 0.9971 | 0.0681 |

在 `eval_cifake_result.json` 中，CIFAKE test set 結果如下：

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9738 |
| Balanced Accuracy | 0.9738 |
| Precision | 0.9575 |
| Recall | 0.9916 |
| F1 | 0.9743 |
| AUROC | 0.9981 |
| Loss | 0.0674 |

這表示模型在 CIFAKE 上能維持非常高的排序能力與召回率，對 fake 圖片的偵測能力強。

### CIFAKE 失真魯棒性

`runs/cifake_no_aug/distortion_results.json` 中的結果如下：

| Distortion | Accuracy | F1 | AUROC |
| --- | ---: | ---: | ---: |
| clean | 0.9720 | 0.9716 | 0.9971 |
| jpeg_q90 | 0.9693 | 0.9687 | 0.9967 |
| jpeg_q70 | 0.9740 | 0.9738 | 0.9969 |
| jpeg_q50 | 0.9475 | 0.9462 | 0.9916 |
| jpeg_q30 | 0.8812 | 0.8667 | 0.9824 |
| blur | 0.7048 | 0.6391 | 0.8155 |
| resize | 0.6653 | 0.5249 | 0.8370 |
| noise | 0.7039 | 0.5805 | 0.9639 |

觀察：

- 輕度 JPEG 壓縮下模型仍維持良好表現。
- 強 JPEG 壓縮會降低 accuracy，但 AUROC 仍相對高。
- blur、resize、noise 對模型影響較大，代表 Branch C 所依賴的局部頻率與紋理訊號容易被這類干擾改變。
- 後續若要提升泛化能力，可考慮在訓練階段加入更完整的 distortion augmentation。

## 設計重點與優勢

### 1. 語意特徵與鑑識特徵互補

Branch A 擅長捕捉整張圖片的語意與高階視覺結構；Branch C 擅長捕捉 patch-level 的局部頻率與紋理異常。兩者融合後，模型不只依賴單一訊號。

### 2. FFT patch selection 提高鑑識分支效率

Branch C 並不是平均看所有 patch，而是根據高頻比例挑出最極端的 high-frequency 與 low-frequency patches。這讓模型更集中在可能包含生成痕跡的位置。

### 3. High / Low 對照設計

只看高頻 patch 可能會過度偏向邊緣與紋理；加入低頻 patch 後，模型可以比較平滑區域與細節區域的差異，增加鑑識訊號的穩定性。

### 4. 可支援跨資料集與跨 generator 評估

資料載入設計支援：

- CIFAKE
- Tiny-GenImage
- CIFAKE + Tiny-GenImage 合併
- Tiny-GenImage 指定 generator
- evaluation by generator

這使得專案可以用於測試模型是否只記住某個生成器，或是否具備跨 domain 泛化能力。

## 可能限制

目前專案仍有幾個可改進方向：

- Branch C 對 blur、resize、noise 等失真較敏感。
- 目前 augmentation 只包含 horizontal flip 與 mild Gaussian blur，對真實社群平台常見壓縮與縮放的覆蓋不足。
- Fusion classifier 使用簡單 feature concatenation，尚未加入 cross-attention 或 gating。
- Branch C 的 patch selection 是固定規則，不是可學習式 patch selection。
- 評估 threshold 固定為 `0.5`，尚未針對不同資料集校準最佳 threshold。

## 後續改進方向

可以考慮以下延伸：

- 加入 JPEG compression、resize、noise、color jitter 等 distortion augmentation。
- 訓練 Branch A only、Branch C only 與 Fusion A+C 的 ablation study。
- 嘗試 ResNet50、ConvNeXt、EfficientNet 或 ViT 作為 Branch A backbone。
- 將 Branch C 的 patch selection 改為 learnable attention 或 differentiable top-k approximation。
- 加入 threshold calibration，針對 validation set 找最佳 F1 或 balanced accuracy threshold。
- 做 cross-dataset evaluation，例如 CIFAKE 訓練、Tiny-GenImage 測試。
- 分析 high-frequency score heatmap，視覺化模型關注的 patch 區域。

## 快速開始

1. 安裝套件：

```bash
pip install -r requirements.txt
```

2. 放置資料集：

```text
dataset/
  cifake/
    train/REAL/
    train/FAKE/
    test/REAL/
    test/FAKE/
```

3. 訓練 CIFAKE fusion model：

```bash
python train.py --config configs/cifake_fusion_a_c.json
```

4. 評估：

```bash
python evaluate.py \
  --checkpoint runs/cifake_fusion_a_c/best.pt \
  --dataset cifake \
  --split test
```

5. 做失真魯棒性測試：

```bash
python evaluate_distortions.py \
  --checkpoint runs/cifake_fusion_a_c/best.pt \
  --model-type fusion \
  --dataset cifake \
  --split test \
  --output-json runs/cifake_fusion_a_c/distortion_results.json \
  --output-csv runs/cifake_fusion_a_c/distortion_results.csv
```

## 總結

本專案實作了一個針對 AI 生成影像偵測的雙分支融合模型。Branch A 透過 ResNet 擷取全域語意特徵，Branch C 透過 FFT patch selection 與 CNN patch encoder 擷取局部鑑識特徵，最後以 MLP 進行 feature-level fusion。實驗結果顯示，模型在 CIFAKE 上能達到高 accuracy、F1 與 AUROC，並且具備完整的訓練、評估、config 管理與魯棒性測試流程。
