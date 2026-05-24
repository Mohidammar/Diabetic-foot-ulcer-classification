# 🧠 HybridVision Classifier

> **EfficientNetV2 + DINOv2 · Siamese Network · Contrastive + CB-Focal Loss · XGBoost Head**

A hybrid deep learning framework that fuses **CNN spatial features** with **Vision Transformer semantics** through similarity learning — designed for fine-grained, class-imbalanced visual classification tasks.

---

## 📌 Overview

| Component | Choice | Role |
|---|---|---|
| CNN Backbone | EfficientNetV2 | Local texture & spatial features |
| ViT Backbone | DINOv2 | Global semantic & structural features |
| Similarity Module | Siamese Network | Metric learning across pairs |
| Loss Function | Contrastive + CB-Focal Fusion | Handles imbalance + pairwise structure |
| Classifier Head | XGBoost | Robust ensemble decision boundary |

The pipeline extracts **dual-stream embeddings**, learns pairwise similarity via a Siamese architecture, and feeds fused representations into a gradient-boosted classifier — achieving strong generalization even under severe class imbalance.

---

## 🏗️ Architecture

```
Input Image
     │
     ├──────────────────────┬──────────────────────┐
     │                      │                      │
EfficientNetV2          DINOv2 ViT             (augmented
(CNN Stream)           (Transformer             pair for
Local features          Stream)                Siamese)
     │                      │
     └──────────┬───────────┘
                │
         Feature Fusion
          (concat / MLP)
                │
        Siamese Network
       (shared weights)
                │
     ┌──────────┴──────────┐
     │                     │
  Embedding A          Embedding B
     │                     │
     └──────────┬───────────┘
                │
    Contrastive + CB-Focal Loss
                │
        XGBoost Classifier
                │
          Prediction
```

---

## ⚙️ Key Components

### 1. Dual-Backbone Feature Extraction

- **EfficientNetV2** — Captures fine-grained local texture, edges, and spatial structure via compound-scaled convolutions.
- **DINOv2** — A self-supervised ViT trained with distillation, capturing rich global semantics and long-range dependencies without label dependency.
- Both embeddings are concatenated (or fused via a lightweight MLP) to form a unified representation.

### 2. Siamese Network

- Shared-weight encoder processes image **pairs** simultaneously.
- Learns a meaningful **embedding space** where similar samples cluster and dissimilar ones repel.
- Particularly effective for few-shot-style generalization and hard-negative mining scenarios.

### 3. Loss Function Fusion

**Contrastive Loss** — Enforces metric structure on the embedding space:
$$\mathcal{L}_{contrastive} = y \cdot D^2 + (1 - y) \cdot \max(m - D, 0)^2$$

**Class-Balanced Focal Loss (CB-Focal)** — Addresses long-tail distributions:
$$\mathcal{L}_{CB\text{-}Focal} = -\frac{1 - \beta}{1 - \beta^{n_y}} \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

Final loss:
$$\mathcal{L} = \alpha \cdot \mathcal{L}_{contrastive} + (1 - \alpha) \cdot \mathcal{L}_{CB\text{-}Focal}$$

### 4. XGBoost Classifier Head

- Extracted embeddings are fed into an **XGBoost** gradient-boosted tree ensemble.
- Avoids softmax over-confidence; handles non-linear decision boundaries in embedding space.
- Supports early stopping, feature importance analysis, and is robust to feature scale differences.

---

## 🔀 Data Augmentation Pipeline

Augmentations are applied per-stream to maximize diversity while preserving label integrity:

### Geometric Transforms
| Transform | Parameters |
|---|---|
| Random Horizontal / Vertical Flip | p = 0.5 |
| Random Rotation | ±15° |
| Random Affine | translate ±10%, scale 0.9–1.1 |
| Random Perspective | distortion scale 0.2 |
| Random Resized Crop | scale (0.7, 1.0), ratio (0.75, 1.33) |

### Color & Texture Transforms
| Transform | Parameters |
|---|---|
| Color Jitter | brightness 0.3, contrast 0.3, saturation 0.2, hue 0.1 |
| Random Grayscale | p = 0.1 |
| Gaussian Blur | kernel (3, 7), sigma (0.1, 2.0) |
| Random Gaussian Noise | std 0.01–0.05 |

### Advanced / Regularization
| Transform | Parameters |
|---|---|
| CutOut / Random Erasing | p = 0.3, scale (0.02, 0.15) |
| MixUp | α = 0.4, applied at batch level |
| CutMix | α = 1.0, applied at batch level |
| AutoAugment / RandAugment | N=2, M=9 |

> **Siamese-specific:** Each branch receives an independently sampled augmentation — the network must learn invariance to these transformations while preserving class identity.

---

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.9
torch >= 2.0
torchvision
timm          # EfficientNetV2 + DINOv2
xgboost
scikit-learn
albumentations
```



## 📄 License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

---

