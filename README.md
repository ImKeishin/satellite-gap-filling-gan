# Sentinel-2 Temporal Gap Filling using GAN

This project implements a Generative Adversarial Network (GAN) for reconstructing cloud-covered regions in multi-temporal Sentinel-2 satellite imagery.

The final pipeline is designed for the **SEN12MS-CR-TS** dataset and uses multispectral optical observations collected at different time points. The model receives multiple cloudy Sentinel-2 images of the same spatial patch and generates a reconstructed 13-band image using temporal context.

The project was developed as the technical component of a bachelor's thesis focused on:

> **Reconstructing missing images from satellite time series using generative models (GANs)**

---

## Overview

Optical satellite observations are frequently affected by clouds, haze, and shadows. These artifacts can hide relevant information about vegetation, water, soil, and urban areas.

The objective of this project is to reconstruct the missing or unreliable regions by exploiting:

* multispectral Sentinel-2 information;
* temporal observations of the same geographical area;
* cloud masks generated automatically with `s2cloudless`;
* a conditional GAN architecture.

The implemented baseline uses:

```text
3 temporal Sentinel-2 images × 13 spectral bands = 39 input channels
```

and generates:

```text
1 reconstructed Sentinel-2 image × 13 spectral bands
```

---

## Dataset

The final pipeline uses the [SEN12MS-CR-TS dataset](https://patricktum.github.io/cloud_removal/sen12mscrts/).

SEN12MS-CR-TS is a multimodal and multitemporal cloud-removal dataset containing:

* 30 observations collected throughout the year 2018 for each region of interest;
* Sentinel-2 optical images with 13 spectral bands;
* Sentinel-1 radar observations;
* cloud-covered and relatively clear observations;
* globally distributed regions of interest;
* separate train and test splits.

This implementation uses **Sentinel-2 only**. Sentinel-1 data is not required for the baseline model.

### Sentinel-2 spectral bands

```text
B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12
```

### Expected extracted structure

The value passed to `--root-dir` must directly contain the `ROIs...` folders:

```text
dataset_root/
  ROIs1970/
    21/
      S2/
        0/
          *.tif
        1/
          *.tif
        ...
        29/
          *.tif
  ROIs2017/
    ...
```

Each GeoTIFF file already contains the 13 Sentinel-2 bands. Separate folders for individual bands are not required.

The dataset is not included in this repository because of its size.

---

## Data Selection Strategy

For every spatial patch, the loader reads the complete sequence of 30 temporal observations.

Cloud masks are computed automatically using `s2cloudless`.

The loader then selects:

* the least-cloudy temporal observation as the reconstruction target;
* cloudier observations as temporal input conditions;
* the union of the selected cloud masks as an auxiliary mask.

The default configuration uses three cloudy temporal inputs:

```text
input shape:  (39, H, W)
target shape: (13, H, W)
mask shape:   (1, H, W)
```

The target is the **least-cloudy available observation**, rather than a guaranteed perfectly cloud-free image.

---

## Model Architecture

The reconstruction system is implemented as a conditional GAN.

### Generator

The generator uses a **U-Net** architecture:

* encoder-decoder structure;
* skip connections for preserving spatial details;
* 39 input channels by default;
* 13 output channels.

### Discriminator

The discriminator uses a conditional **PatchGAN** architecture.

It receives:

```text
temporal condition + real or generated multispectral image
```

and evaluates the realism of local image patches.

### Loss Function

The generator loss combines:

```text
generator_loss = adversarial_loss + lambda_L1 × reconstruction_loss
```

where:

* adversarial loss encourages realistic reconstructions;
* L1 reconstruction loss encourages similarity to the target;
* `lambda_L1 = 100` by default.

The discriminator is trained to distinguish real targets from generated images.

---

## Evaluation Metrics

The reconstruction quality is evaluated using:

* **MAE** — Mean Absolute Error;
* **PSNR** — Peak Signal-to-Noise Ratio;
* **SSIM** — Structural Similarity Index Measure.

The pipeline also generates RGB visual comparisons containing:

```text
cloudy input | GAN reconstruction | least-cloudy target
```

The complete generated reconstruction is saved separately as a 13-band NumPy array.

---

## Project Structure

```text
satellite-gap-filling-gan/
  scripts/
    train_sen12mscrts.py
    evaluate_sen12mscrts.py
    inference_sen12mscrts.py
    test_sen12mscrts_loader.py
    rank_test_samples.py
    download_sen12mscrts_official.sh

  src/
    dataset/
      sen12mscrts_loader.py

    models/
      generator.py
      discriminator.py
      gan.py

    training/
      trainer.py

    utils/
      metrics.py
      visualization.py

  notebooks/
  requirements.txt
  README.md
  LICENSE
```

The repository also contains scripts used during the initial development stage with a simplified `.npy` dataset. The final SEN12MS-CR-TS pipeline is implemented in the files whose names contain `sen12mscrts`.

---

## Installation

Create and activate a Python virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux or WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main dependencies:

```text
torch
torchvision
numpy
rasterio
s2cloudless
scikit-image
matplotlib
tqdm
```

CUDA-compatible hardware is recommended for training.

---

## Dataset Download

The official dataset can be downloaded from the [SEN12MS-CR-TS project page](https://patricktum.github.io/cloud_removal/sen12mscrts/).

The dataset is split by sensor and geographical region. For a smaller local experiment, only Sentinel-2 data for Africa can be downloaded.

Example extraction commands in WSL:

```bash
mkdir -p train
mkdir -p test

tar --touch --no-same-owner --no-same-permissions \
  -xzf s2_africa.tar.gz \
  -C train

tar --touch --no-same-owner --no-same-permissions \
  -xzf s2_africa_test.tar.gz \
  -C test
```

After extraction, identify the folder that directly contains the `ROIs...` directories and use it as `--root-dir`.

---

## Verify the Dataset Loader

Run a fast smoke test using the lightweight heuristic cloud detector:

```powershell
python -m scripts.test_sen12mscrts_loader `
  --root-dir <path-to-train-root> `
  --split train `
  --region africa `
  --n-input-times 3 `
  --max-samples 2 `
  --cloud-detector heuristic
```

Expected tensor shapes:

```text
masked_input: (1, 39, 256, 256)
target:       (1, 13, 256, 256)
mask:         (1, 1, 256, 256)
```

For the final experiments, use:

```text
--cloud-detector s2cloudless
```

---

## Training

### Smoke Test

Run a short experiment before starting the final training:

```powershell
python -m scripts.train_sen12mscrts `
  --root-dir <path-to-train-root> `
  --region africa `
  --epochs 1 `
  --batch-size 1 `
  --n-input-times 3 `
  --base-channels 16 `
  --max-train-samples 20 `
  --max-val-samples 10 `
  --cloud-detector heuristic `
  --checkpoint-dir checkpoints_sen12mscrts
```

### Final Training Configuration

```powershell
python -m scripts.train_sen12mscrts `
  --root-dir <path-to-train-root> `
  --region africa `
  --epochs 20 `
  --batch-size 2 `
  --n-input-times 3 `
  --base-channels 32 `
  --max-train-samples 3000 `
  --max-val-samples 256 `
  --cloud-detector s2cloudless `
  --checkpoint-dir checkpoints_sen12mscrts_final
```

Generated files:

```text
checkpoints_sen12mscrts_final/
  best.pt
  last.pt
  history.json
```

`best.pt` stores the checkpoint with the lowest validation L1 error.

---

## Evaluation

Evaluate the final checkpoint on the separate test split:

```powershell
python -m scripts.evaluate_sen12mscrts `
  --root-dir <path-to-test-root> `
  --checkpoint checkpoints_sen12mscrts_final\best.pt `
  --split test `
  --region africa `
  --n-input-times 3 `
  --base-channels 32 `
  --max-samples 256 `
  --max-visuals 20 `
  --cloud-detector s2cloudless `
  --output-dir results_sen12mscrts\final_test
```

Generated files:

```text
results_sen12mscrts/
  final_test/
    metrics.csv
    sample_0000.png
    sample_0001.png
    ...
```

The terminal prints the average MAE, PSNR, and SSIM values.

---

## Inference

Generate a 13-band reconstruction and an RGB visual comparison for one test sample:

```powershell
python -m scripts.inference_sen12mscrts `
  --root-dir <path-to-test-root> `
  --checkpoint checkpoints_sen12mscrts_final\best.pt `
  --split test `
  --region africa `
  --sample-index 0 `
  --n-input-times 3 `
  --base-channels 32 `
  --cloud-detector s2cloudless `
  --output-dir results_sen12mscrts\final_inference
```

Generated files:

```text
results_sen12mscrts/
  final_inference/
    prediction_13bands.npy
    comparison.png
```

---

## Rank Easy, Medium, and Difficult Cases

The test samples can be ranked according to cloud coverage.

The ranking script calculates a difficulty score based on the union of the cloud masks from the selected temporal inputs.

```powershell
python -m scripts.rank_test_samples `
  --root-dir <path-to-test-root> `
  --split test `
  --region africa `
  --n-input-times 3 `
  --cloud-detector s2cloudless `
  --max-samples 256 `
  --output-dir results_sen12mscrts\sample_ranking
```

Generated files:

```text
results_sen12mscrts/
  sample_ranking/
    sample_difficulty.csv
    easy_sample_XXXX.png
    medium_sample_XXXX.png
    hard_sample_XXXX.png
```

---

## Experimental Baseline

The final baseline experiment uses:

```text
Dataset:              SEN12MS-CR-TS
Sensor:               Sentinel-2 only
Region:               Africa
Temporal inputs:      3
Input channels:       39
Output channels:      13
Generator:            U-Net
Discriminator:        PatchGAN
Cloud detector:       s2cloudless
Loss:                 adversarial + 100 × L1
Training epochs:      20
Batch size:           2
Training subset:      up to 3000 samples
Validation subset:    up to 256 samples
Evaluation metrics:   MAE, PSNR, SSIM
```

---

## Notes

* The dataset is intentionally excluded from Git.
* Checkpoints and generated results are local artifacts and should not be committed.
* The heuristic cloud detector is intended only for quick tests.
* The final experiments should use `s2cloudless`.
* The complete multispectral predictions are saved as 13-band `.npy` arrays.
* RGB images are generated only for visualization.

---

## References

The project uses the SEN12MS-CR-TS dataset:

```text
Ebel, P., Xu, Y., Schmitt, M., and Zhu, X. X.
SEN12MS-CR-TS: A Remote Sensing Data Set for Multi-modal Multi-temporal Cloud Removal.
IEEE Transactions on Geoscience and Remote Sensing, 2022.
```

The cloud masks are generated using:

```text
Sentinel Hub s2cloudless:
Sentinel-2 cloud detector for Python.
```

---

## License

This project is distributed under the MIT License.
