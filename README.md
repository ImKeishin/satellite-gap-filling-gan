# Satellite Gap Filling using GAN

This project focuses on reconstructing missing regions in Sentinel-2 multispectral satellite image time series using Generative Adversarial Networks (GANs).

---

## Overview

Satellite images often contain missing data due to clouds, shadows, or sensor limitations.
The goal of this project is to reconstruct these missing regions while preserving spatial structure and temporal consistency.

---

## Key Features

* Reconstruction of missing regions caused by clouds and shadows
* Use of multispectral Sentinel-2 data
* GAN-based model (Generator + Discriminator)
* Patch-based processing of images
* Evaluation using PSNR, SSIM, and MAE

---

## Dataset

* Sentinel-2 multispectral data
* Stored as `.npy` files
* Each file represents a temporal stack for a spectral band: `(H, W, T)`

The dataset is not included in this repository.

---

## Model

* Generator: U-Net architecture
* Discriminator: PatchGAN
* Loss function:

  * Adversarial loss
  * L1 reconstruction loss

---

## Project Structure

* `data/` – dataset (not included)
* `src/` – model, dataset loader, utilities
* `scripts/` – training and preprocessing scripts
* `results/` – generated images and evaluation results
* `notebooks/` – data exploration and analysis

---

## Evaluation

The model is evaluated using:

* PSNR
* SSIM
* MAE

and visual comparison between:

* input image (with missing regions)
* reconstructed output
* ground truth

---

## Goal

To build a model capable of reconstructing missing satellite data, with potential applications in agriculture and environmental monitoring.
