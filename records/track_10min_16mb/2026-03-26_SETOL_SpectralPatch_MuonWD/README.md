# SETOL / WeightWatcher Spectral Patch (Muon WD)

This record implements a comprehensive spectral diagnostics suite for the **Parameter Golf** 10-minute/16MB track. By monitoring the empirical spectral density (ESD) of weight matrices during training, we achieve more stable convergence and better post-training quantization.

## Key Features

1.  **SpectralMonitor / Convergence Oracle**:
    - Tracks the mean tail exponent (ᾱ) and inter-layer variance (σ²_α).
    - Automatically identifies the "heavy-tail" regime (ᾱ ∈ [2.2, 3.8]).
    - Signals the onset for Quantization-Aware Training (QAT) when weights have sufficiently differentiated.
    - Triggers early warmdown when spectral stability is reached.

2.  **AlphaDecay (Muon Optimization)**:
    - Scales weight decay (WD) per-parameter based on the layer's ᾱ.
    - Layers with light tails (higher ᾱ) receive more WD to push them toward the critical regime.
    - Near-critical layers (lower ᾱ) are protected with minimal WD to preserve learned structure.

3.  **α-Selective Quantization**:
    - Dynamically allocates bit-widths per transformer block.
    - Blocks with near-critical layers use standard `int8` to prevent information loss.
    - Blocks with high ᾱ (diffuse information) are rounded to `int6` (via `INT6_STEP=4`) for better `zlib` compression.

4.  **Stable-Rank GPTQ**:
    - Optimizes the quantization clipping threshold (grid search) using a Hessian diagonal approximation.
    - Clipping percentiles are informed by the layer's **stable rank**. Low-rank layers receive conservative clipping to protect their sparse information density.

## Usage

Ensure `weightwatcher` is installed:
```bash
pip install weightwatcher --break_system-packages
```

Run training:
```bash
python train_gpt.py
```

## Theory

This integration is based on Heavy-Tailed Self-Regularization (HTSR) theory (Martin & Mahoney). The training loop aims to maintain the model at the "Edge of Stability," where the ESD of weight matrices follows a power law with α ≈ 2. This regime minimizes generalization error and maximizes the efficiency of the 16MB int8/int6 representation.
