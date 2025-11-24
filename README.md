# Dual-Stream Structural Representation Learning for Structure-Rich Sequential Signals  
**(Code will be released upon acceptance of the paper.)**

This repository hosts the official implementation of a unified **dual-stream representation learning framework** designed for heterogeneous one-dimensional signals, including **near-infrared (NIR) spectroscopy**, **audio waveforms**, and **electrocardiograms (ECG)**. The framework integrates a *learnable 1D-to-2D structural transform*, cross-modal semantic alignment, and reliability-aware feature fusion to model both **local temporal dynamics** and **global structural dependencies**.

---

## Overview

Traditional 1D encoders (CNNs, RNNs, Transformers) effectively capture short-range temporal variations but often struggle to represent long-range relational structures that play a critical role in time-series interpretation. Conversely, existing handcrafted 1D-to-2D projections (GAF, MTF, RP) introduce fixed geometric biases and cannot adapt to modality-specific characteristics or noise distributions.

This work proposes a **unified, learnable, and task-adaptive solution** that addresses these limitations through four key components:

### 1. Learnable Spectral–Image Transform (LSIT)
A differentiable structural reconstruction module that models:
- Positional continuity  
- Nonlinear morphological similarity  
- Global correlation trends  
- Gradient-level coherence  

LSIT learns a modality-adaptive 2D relational map that generalizes beyond handcrafted projections.

### 2. Local Representation Encoder (LRE)
A lightweight temporal encoder based on multi-scale 1D convolutions and self-attention, responsible for capturing:
- Fine-grained temporal transitions  
- High-frequency spectral fluctuations  
- Short-range morphological patterns  

### 3. Dynamic Cross-Attention (DCA)
A bidirectional interaction mechanism enabling:
- Local queries of global structure  
- Global queries of local evidence  
- Adaptive temperature scaling  
- Top-k sparsification to suppress noise  

This module enforces semantic alignment across the two streams.

### 4. Dynamic Gated Fusion (DGF) + Feature Alignment Loss (FAL)
A reliability-aware fusion mechanism that:
- Learns soft mixture coefficients  
- Minimizes cross-stream redundancy  
- Provides feedback signals to refine LSIT  
- Enhances robustness under drift, noise, and domain shifts  

---

## Key Features

- **Universal applicability** across NIR, audio, and ECG modalities  
- **Learnable 1D-to-2D mapping** instead of handcrafted transformations  
- **Cross-scale reasoning** between temporal and structural views  
- **Robustness** to baseline drift, SNR degradation, and physiological distortions  
- **Efficient computation** suitable for portable and embedded devices  
- **Interpretable structural representations** via relational kernel mixtures  

---

## Experimental Results

The proposed framework achieves **consistent state-of-the-art performance** across:

### NIR Spectroscopy
- Mango and Tree Organs datasets  
- Outperforms 1D CNNs, hybrid models, and GAF/MTF/RP-based approaches  

### Audio Classification
- UrbanSound8K and AudioMNIST  
- Surpasses wavLM, HuBERT, Data2Vec, neural codec models, and strong baselines  

### ECG Classification
- CHB-MIT and PTB-XL  
- Exceeds SpaRCNet, BIOT, SimMTM, and CHi-MTS  

Robustness evaluations show strong stability under drift, amplitude corruption, and noisy environments.

---

## Repository Structure  
*(Will be updated upon code release)*


---

## Citation

A BibTeX template will be provided once the paper is accepted and published.

---

## License

The codebase will be released under a permissive academic license upon acceptance.

---

## Contact

For any questions regarding implementation or experimental setup, please open an issue once the repository becomes public.

---
