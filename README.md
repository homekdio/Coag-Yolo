# Coag-YOLOv7: YOLOv7 with CoTAttention and CARAFE

**Coag-YOLOv7** is an improved object detection model based on [YOLOv7](https://github.com/WongKinYiu/yolov7), designed for **fine-grained crop pest and disease detection**. It integrates two plug-and-play modules — **CoTAttention** and **CARAFE** — to enhance feature extraction and upsampling quality.

## ✨ Key Improvements

| Module | Location | Purpose |
|---|---|---|
| **CoTAttention** | End of Backbone (after Layer 50) | Enhances global contextual awareness by fusing static context keys with dynamically weighted values |
| **CARAFE** ×2 | FPN Upsampling (replaces `nn.Upsample`) | Content-aware upsampling that preserves semantic details using feature-guided reassembly kernels |

### Architecture Overview

```
                        ┌─────────────────────────────┐
                        │        Backbone              │
                        │  (Same as YOLOv7 Layers 0-50)│
                        └──────────┬──────────────────┘
                                   │
                        ┌──────────▼──────────────────┐
                        │  ✅ CoTAttention (Layer 51)  │
                        │  Global context enhancement  │
                        └──────────┬──────────────────┘
                                   │
                        ┌──────────▼──────────────────┐
                        │       SPPCSPC (Layer 52)     │
                        └──────────┬──────────────────┘
                                   │
                    ┌──────────────┼──────────────────┐
                    │              │                    │
             ┌──────▼──────┐ ┌────▼────────┐   ┌──────▼──────┐
             │ ✅ CARAFE   │ │  ✅ CARAFE  │   │  Downsample │
             │ P4 Upsample │ │ P3 Upsample │   │    Path     │
             └──────┬──────┘ └────┬────────┘   └──────┬──────┘
                    │              │                    │
             ┌──────▼──────┐ ┌────▼────────┐   ┌──────▼──────┐
             │  P4 Fusion  │ │  P3 Fusion  │   │  P5 Fusion  │
             └──────┬──────┘ └────┬────────┘   └──────┬──────┘
                    │              │                    │
                    └──────────────┼────────────────────┘
                                   │
                            ┌──────▼──────┐
                            │   IDetect   │
                            │  (P3,P4,P5) │
                            └─────────────┘
```

## 📊 Results

Evaluated on a custom crop pest and disease dataset with **22 classes**.

### Comparison Experiment

| Algorithm | mAP@0.5 (%) | Recall (%) | Params (M) | FPS |
|---|---|---|---|---|
| Faster-RCNN | 82.9 | 89.6 | 80.4 | 15.1 |
| SSD | 81.6 | 81.3 | 48.3 | 75.4 |
| YOLOv3 | 76.3 | 79.2 | 68.9 | 57.2 |
| YOLOv5(m) | 83.7 | 75.8 | 38.2 | 110.6 |
| YOLOv7 | 84.3 | 71.3 | 37.6 | 117.8 |
| **Coag-YOLOv7** | **89.4** | **77.1** | **47.0** | **108.3** |

### Ablation Study

| Model | mAP@0.5 (%) | Recall (%) | Params (M) | FPS |
|---|---|---|---|---|
| YOLOv7 (Baseline) | 84.3 | 71.3 | 37.6 | 117.8 |
| + CoTAttention | 87.1 (+2.8) | 74.5 (+3.2) | 46.8 (+9.2) | 109.5 (-8.3) |
| + CARAFE | 86.4 (+2.1) | 73.7 (+2.4) | 37.8 (+0.2) | 114.2 (-3.6) |
| **Coag-YOLOv7 (Both)** | **89.4 (+5.1)** | **77.1 (+5.8)** | **47.0 (+9.4)** | **108.3 (-9.5)** |

## 🚀 Getting Started

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.12
- CUDA (recommended for training)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/coag-yolov7.git
cd coag-yolov7
pip install -r requirements.txt
```

### Training

Train with the Coag-YOLOv7 architecture:

```bash
python train.py \
    --workers 8 \
    --device 0 \
    --batch-size 16 \
    --epochs 300 \
    --img 640 640 \
    --data data/your_dataset.yaml \
    --hyp data/hyp.scratch.p5.yaml \
    --cfg cfg/training/yolov7-coag.yaml \
    --weights yolov7.pt \
    --name coag-yolov7
```

> **Tip**: You can use the original YOLOv7 pretrained weights (`yolov7.pt`) for transfer learning. Mismatched layers (CoTAttention, CARAFE) will be randomly initialized automatically.

### Inference

```bash
python detect.py \
    --weights runs/train/coag-yolov7/weights/best.pt \
    --source your_image.jpg \
    --img-size 640 \
    --conf-thres 0.25
```

### Testing

```bash
python test.py \
    --weights runs/train/coag-yolov7/weights/best.pt \
    --data data/your_dataset.yaml \
    --img 640 \
    --batch-size 32 \
    --task test
```

## 📁 Modified Files

Compared to the original [YOLOv7](https://github.com/WongKinYiu/yolov7), the following files are modified:

| File | Change |
|---|---|
| `models/common.py` | Added `CoTAttention` and `CARAFE` class definitions |
| `models/yolo.py` | Registered new modules in `parse_model()` |
| `cfg/training/yolov7-coag.yaml` | New config with CoTAttention + CARAFE architecture |

### CoTAttention (Contextual Transformer Attention)

- **Paper**: [Contextual Transformer Networks for Visual Recognition (2021)](https://arxiv.org/abs/2107.12292)
- Fuses static context (group convolution keys) with dynamic attention (softmax-weighted values)
- Inserted at the **end of backbone** to capture global long-range dependencies before entering the FPN neck

### CARAFE (Content-Aware ReAssembly of Features)

- **Paper**: [CARAFE: Content-Aware ReAssembly of FEatures (ICCV 2019)](https://arxiv.org/abs/1905.02188)
- Predicts per-pixel upsampling kernels from the feature map itself
- Replaces naive nearest-neighbor upsampling in FPN, preserving fine-grained semantic details
- Near-zero parameter overhead (+0.2M per module)

## 📄 License

This project inherits the [GPL-3.0 License](LICENSE.md) from the original YOLOv7 repository.

## 🙏 Acknowledgements

- [YOLOv7](https://github.com/WongKinYiu/yolov7) — Base detection framework
- [CoTNet](https://github.com/JDAI-CV/CoTNet) — Contextual Transformer Attention
- [CARAFE](https://github.com/open-mmlab/mmdetection) — Content-Aware ReAssembly of Features
