<h1 align="center">QAD</h1>
<p align="center">
  <strong>Quantization Adversarial Distillation for Embedded Underwater Acoustic Target Recognition</strong><br>
  Submitted to <strong>IEEE Embedded Systems Letters(ESL) 2026</strong>
</p>

## Introduction
<p align="center">
  <img src="./assets/QAD Figure 1.png" width="800px">
</p>

To address the challenges of resource-constrained UATR sensing platforms, we propose **Quantization Adversarial Distillation (QAD)**, a novel framework that integrates **knowledge distillation (KD)** and **quantization-aware training (QAT)** with **Adversarial Training** to achieve both significant model compression and enhanced recognition performance.

**Our contributions are below.

** 1. Quantization-Adversarial Distillation : Restore the information loss during compaction model size (V3 to V1)
** 2. Inferenece Latency - Accuracy Trade-off : we acheived that reduce inference latency and increase accuracy at QAD V3 compare to the full-precision model. Reducing the model size to 37% and inference latency to 44% compared to the full-precision baseline model.
** 3. Deployment on the Raspberry Pi 5 : Our work measure the performance at real edge device "Rapberry Pi 5", accuracy and latency.**

## Performance Benchmarking of CNN-based Models and Transformer
<p align="center">
  <img src="./assets/compare_models.png" width="800px">
</p>

## Evalutation Summary Across The Metrics
<p align="center">
  <img src="./assets/Figure 2.png" width="500px">
</p>

Our CQTF (Computation-Quantized Training Framework) implementation is heavily based on and adapted from the official repository: [Xingzhi-Zhou/CQTF_PyTorch](https://github.com/Xingzhi-Zhou/CQTF_PyTorch).

## Installation
1. Clone this repository and go to QAD folder
```bash
git clone https://github.com/KNU-LMAP/edge-qad.git
cd edge-qad/
```
2. Create a conda environment and install requirements
```bash
conda env create -f environment.yaml
conda activate QAD
```
## Dataset Preprocessing
1. To reproduce the results, ensure your dataset follows the **Directory Structure** below.
```text
DeepShip/
├── Cargo/
│   ├── 1.wav
│   ├── 2.wav
│   └── ...
├── Passengership/
│   ├── 1.wav
│   ├── 2.wav
│   └── ...
├── Tug/
│   ├── 1.wav
│   ├── 2.wav
│   └── ...
└── Tanker/
    ├── 1.wav
    ├── 2.wav
    └── ...
```
2. Go to data/ directory.
3. Run split_data.py. (Ensure you set the raw_data_root and output_root inside the script or via arguments.)
```bash
cd data/
python split_data.py
```

## Train
1. Go to scripts/ directory
```bash
cd ../scripts/
```
2. Open the desired .sh file and set your data_root to the preprocessing output root.
3. Run *.sh
```bash
chmod +x *.sh
./train_qad.sh
./train_kd.sh
./train_qat.sh
./train_ref.sh
./train_ad.sh
```
## Citation
