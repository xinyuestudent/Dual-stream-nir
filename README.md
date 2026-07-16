# LSIT Multimodal Modules

Official modular implementation for **Consistency-Aware Structural Modeling of Time-Series Signals via Learnable Spectral-Image Transform**.

This repository provides a PyTorch implementation of **Learnable Spectral-Image Transform (LSIT)** and a dual-stream classification framework for robust time-series analysis. The code was modularized from `lsit-multimodalv2-normalization-new.ipynb` and supports spectral, audio, ECG, and EEG-style inputs.

## Overview

Time-series signals often contain two complementary types of information:

- local waveform patterns, such as short-range spectral variations, acoustic transients, ECG morphology, or EEG fluctuations;
- global structural dependencies, such as long-range correlations, periodicity, baseline trends, and cross-position relationships.

Conventional 1D models capture local patterns effectively but often model long-range structure only implicitly. Fixed signal-to-image transformations such as GAF, MTF, and RP can expose pairwise relationships, but their predefined mapping rules are not adaptive to different signal conditions.

LSIT addresses this limitation by learning a structural relation map directly from raw 1D signals. The resulting 2D structural representation is processed by an image branch and fused with local temporal features through dynamic cross-attention and dynamic gated fusion.

## Main Features

- Learnable Spectral-Image Transform (LSIT) for adaptive structural modeling.
- Dual-stream classifier combining a 1D spectral branch and a 2D structural-image branch.
- Built-in comparison transforms: `lsit`, `gaf`, `mtf`, and `rp`.
- Dynamic Cross-Attention (DCA) for local-global feature interaction.
- Dynamic Gated Fusion (DGF) and fusion alignment losses.
- Training, evaluation, model summary, confusion matrix, and visualization utilities.
- Preprocessing helpers for spectral CSV data, AudioMNIST, UrbanSound8K, MIT-BIH-style ECG, and EEG CSV inputs.

## File Structure

```text
lsit_multimodal_modules/
|-- augment.py           # Spectral augmentation utilities
|-- cross_attention.py   # Dynamic Cross-Attention module
|-- dual_stream.py       # DualStreamNIRNet classifier
|-- image_branch.py      # 2D structural-image feature extractor
|-- models.py            # Backward-compatible model exports
|-- preprocessing.py     # Spectral, audio, ECG, and EEG preprocessing/loaders
|-- run_example.py       # Minimal command-line training example
|-- spectral_branch.py   # 1D local representation encoder
|-- train_eval.py        # Training, evaluation, metrics, summaries, confusion matrix
|-- transforms.py        # LSIT, GAF, MTF, and RP transforms
|-- visualization.py     # t-SNE/PCA, LSIT metrics, feature and confidence plots
`-- __init__.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Create an environment:

```bash
conda create -n lsit python=3.9
conda activate lsit
```

Install dependencies:

```bash
pip install torch torchvision torchaudio numpy pandas scipy scikit-learn matplotlib tqdm librosa thop
```

If your repository includes `requirements.txt`, you can use:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the example training pipeline from the parent directory of `lsit_multimodal_modules`:

```bash
python -m lsit_multimodal_modules.run_example \
  --csv spectral_data.csv \
  --label-col label \
  --num-classes 2 \
  --input-length 400 \
  --batch-size 32 \
  --epochs 50 \
  --transform lsit \
  --save-path checkpoints/best_model.pt
```

Available transform choices:

```text
lsit, gaf, mtf, rp
```

The example script trains the model, evaluates classification metrics, saves the best checkpoint, extracts fused features, and generates a t-SNE visualization.

## Python Usage

```python
import torch

from lsit_multimodal_modules.preprocessing import load_spectral_csv
from lsit_multimodal_modules.models import DualStreamNIRNet
from lsit_multimodal_modules.train_eval import train, evaluate_metrics

bundle = load_spectral_csv(
    "spectral_data.csv",
    label_col="label",
    batch_size=32,
    crop=(800, 1200),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualStreamNIRNet(
    input_shape=bundle.train_data.shape[1],
    num_classes=2,
    transform="lsit",
)
model.to(device)

# DualStreamNIRNet uses LazyLinear layers in branch heads, so run one
# dummy forward pass before creating the optimizer.
with torch.no_grad():
    model(torch.randn(1, bundle.train_data.shape[1], device=device))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

train(
    model,
    bundle.train_loader,
    bundle.test_loader,
    criterion,
    optimizer,
    num_epochs=50,
    device=device,
    save_path="checkpoints/best_model.pt",
    name="lsit",
)

metrics = evaluate_metrics(
    model,
    bundle.test_loader,
    criterion,
    device,
    save_dir="eval_results",
)
print(metrics)
```

## Data Loading

The preprocessing module includes loaders and helpers for several input types:

- `load_spectral_csv`: spectral CSV loading, correction, smoothing, derivative preprocessing, cropping, and train/test splitting.
- `load_audio_mnist`: AudioMNIST folder loading with MFCC extraction.
- `load_urbansound8k`: UrbanSound8K metadata-based loading with MFCC extraction.
- `load_mitbih`: MIT-BIH-style ECG CSV loading.
- `load_eeg_csv`: EEG CSV loading.
- `make_loaders` and `tensors_to_loaders`: utilities for custom arrays or tensors.

For spectral CSV data, the default label column is `label`. Keep `input_shape` equal to the post-preprocessing sequence length, for example the length after `crop=(800, 1200)`.

## Model Components

The core model is `DualStreamNIRNet`:

1. **SpectrumModel**
   - Local 1D branch for waveform or spectral feature extraction.

2. **LSIT / GAF / MTF / RP**
   - Structural transformation branch. LSIT is learnable, while GAF, MTF, and RP are available as fixed transformation baselines.

3. **ImageBranch**
   - 2D encoder for the structural relation map.

4. **DynamicCrossAttention**
   - Cross-scale alignment between local and global representations.

5. **Dynamic Gated Fusion**
   - Adaptive fusion of spectral, structural, and cross-attended features.

## Outputs

The example workflow can generate:

- trained checkpoints in `checkpoints/`;
- evaluation outputs in `eval_results/`;
- t-SNE visualizations in `lsit_visuals/`;
- confusion matrices and classification metrics from `train_eval.py`;
- distance, confidence, centroid, and LSIT activation reports from `visualization.py`.

## Reported Paper Results

The paper evaluates the proposed framework on speech, ECG, and EEG benchmarks:

| Dataset      | Task                               | Reported Performance          |
| ------------ | ---------------------------------- | ----------------------------- |
| AudioMNIST   | Speech classification              | 99.71% accuracy               |
| UrbanSound8K | Environmental sound classification | Competitive performance       |
| PTB-XL       | ECG classification                 | 92.1% accuracy                |
| CHB-MIT      | EEG classification                 | Consistent robust performance |

Robustness analysis shows stable performance under perturbations. On AudioMNIST, accuracy decreases from 0.9971 under clean conditions to 0.9621 under 5 dB AWGN. On PTB-XL, accuracy decreases from 0.921 at 0.1 Hz baseline drift to 0.888 at 0.3 Hz drift.

## Efficiency

Under the unified settings reported in the paper, the model has:

| Model             | Parameters | Model Size | BFLOPs | Average Forward Time |
| ----------------- | ---------: | ---------: | -----: | -------------------: |
| Ours without LSIT |     3.85 M |   14.70 MB |   3.10 |              12.9 ms |
| Ours              |     4.03 M |   15.38 MB |   3.51 |              13.8 ms |

LSIT adds a small computational overhead while improving structural consistency and classification performance.

## Citation

If this code is useful for your research, please cite:

```bibtex
@article{wang2026lsit,
  title   = {Consistency-Aware Structural Modeling of Time-Series Signals via Learnable Spectral-Image Transform},
  author  = {Wang, Xinyue and Chen, Xiangdong},
  journal = {To be updated},
  year    = {2026}
}
```



## Acknowledgments

This work was supported by the Fundamental Research Funds for the Central Universities under Grant 2682022ZTPY001.

## Contact

For questions or suggestions, please contact:

- Xinyue Wang: wangxinyuecins@gmail.com
