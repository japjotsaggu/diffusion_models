# Generative Models: Hot Dog Generator

A deep learning coursework project implementing generative models from scratch, culminating in a **Latent Diffusion Model (LDM)** capable of generating hot dog images.

Note: This repository contains my coursework submission for the Deep Learning module (70010 | Faculty of Engineering) at Imperial College London.
---

## Overview

This project is split into two main parts:

1. **Part 1 — Variational Autoencoder (VAE):** Train a convolutional VAE on MNIST and then adapt it to generate hot dog images.
2. **Part 2 — Denoising Diffusion Probabilistic Model (DDPM):** Build and train a latent diffusion model on top of Stable Diffusion's pretrained VAE to generate plausible hot dog images.

The project is inspired by the hot dog detector from *Silicon Valley* and uses a curated hot dog dataset. Generated images are evaluated against a ResNet-based food classifier, with success defined as achieving a top-5 prediction of "hot dog".

---

## Part 1: Variational Autoencoder

### Architecture
A convolutional VAE (<1M parameters) with:
- **Encoder:** 3× Conv2d blocks (32→64→128 channels) + BatchNorm + ReLU, followed by linear layers to output `μ` and `log σ²`
- **Reparameterisation trick:** `z = μ + ε · σ`, where `ε ~ N(0, I)`
- **Decoder:** Linear projection → 3× ConvTranspose2d blocks → Sigmoid output

### Training
- **MNIST:** 16 epochs, BCE reconstruction loss + β-weighted KL divergence (`β=5`), Adam optimiser with step-LR scheduler
- **Hot dogs:** 60 epochs, MSE (Gaussian NLL) reconstruction loss, same setup adapted for 3-channel RGB images at 28×28

### Key Findings
- Latent space interpolations reveal smooth transitions between MNIST digits; hot dog interpolations are visually coherent but lack fine detail
- T-SNE visualisations (perplexity 5, 30, 50) show reasonable digit class separation in the learned latent space

---

## Part 2: Latent Diffusion Model

### Setup
Rather than training a VAE from scratch on hot dogs (which yields low quality at limited compute), the pretrained **Stable Diffusion 2 VAE** from HuggingFace is used as the encoder/decoder backbone. The DDPM operates entirely in this compressed latent space (4×14×14 for 112×112 inputs).

### DDPM Schedules
Linear noise schedule `β₁=1e-4` → `β₂=0.02` over `T=1000` steps, with precomputed `ᾱ_t`, `√ᾱ_t`, `√(1−ᾱ_t)`, and posterior variance terms.

### Noise Predictor: SimpleUnet
A U-Net with sinusoidal timestep positional embeddings:
- **Sinusoidal embeddings** encode the diffusion timestep and are injected into each block via an MLP projection
- **Encoder path:** 128→256→512 channels with skip connections stored
- **Bottleneck:** 512 channels
- **Decoder path:** skip-concatenation + upsampling via ConvTranspose2d
- ~16M parameters, well within the 20M limit

### Training
- Latent codes are normalised using global dataset statistics (mean/std computed once before training)
- Loss: MSE between predicted and true noise `‖ε − ε_θ(x_t, t)‖²`
- Optimiser: Adam, `lr = 2e-4 × batch_size`
- 150 epochs; checkpoints and best-generated images saved per epoch

### Evaluation
Generated images are scored by a ResNet food classifier. Success = hot dog appearing in the top-5 predictions. Best generated image and confidence score are tracked and saved during training.

---

## References

- [DDPM Paper — Ho et al., 2020](https://arxiv.org/abs/2006.11239)
- [Lilian Weng — Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Lilian Weng — VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)
- [Stable Diffusion VAE — HuggingFace](https://huggingface.co/Manojb/stable-diffusion-2-base)
- [Hot Dog Dataset — Roboflow](https://universe.roboflow.com/workspace-2eqzv/hot-dog-detection/dataset/2)
