# Quantization Adversarial Distillation for Embedded Underwater Acoustic Target Recognition
QAD, Submitted to **IEEE Geoscience and Remote Sensing Letters(GRSL) 2026**

## Introduction
<p align="center">
  <img src="./assets/QAD Figure 1.png" width="800px">
</p>

To address the challenges of resource-constrained UATR sensing platforms, we propose **Quantization Adversarial Distillation (AQD)**, a novel framework that integrates knowledge distillation with quantization-aware training to achieve both significant model compression and enhanced recognition performance.

## Evalutation Summary Across The Metrics
<div align="center">
  
| Strategy | dtype | Acc (%) | MACs | Size (KB) | Inf. $^\dagger$ (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FAC (Teacher)** | | | | | |
| Reference | FP32 | 72.69 ± 0.86 | 129.45 M | 9602 | 16.97 ± 0.37 |
| **ShuffleFAC V3** | | | | | |
| Reference | FP32 | 71.64 ± 1.92 | 9.85 M | 636 | 7.50 ± 1.43 |
| SKD | FP32 | 72.40 ± 1.01 | 9.85 M | 636 | 7.50 ± 1.43 |
| AD | FP32 | 72.73 ± 1.24 | 9.85 M | 636 | 7.50 ± 1.43 |
| QAT | INT8 | 73.59 ± 0.24 | 9.85 M | 238 | 4.67 ± 0.44 |
| **QAD (Ours)** | **INT8** | **74.11 ± 0.28** | **9.85 M** | **238** | **4.67 ± 0.44** |
| **ShuffleFAC V2** | | | | | |
| Reference | FP32 | 71.85 ± 0.85 | 3.06 M | 226 | 6.23 ± 1.20 |
| SKD | FP32 | 72.21 ± 1.59 | 3.06 M | 226 | 6.23 ± 1.20 |
| AD | FP32 | 72.52 ± 0.77 | 3.06 M | 226 | 6.23 ± 1.20 |
| QAT | INT8 | 71.81 ± 0.40 | 3.06 M | 113 | 4.46 ± 0.44 |
| **QAD (Ours)** | **INT8** | **73.22 ± 0.26** | **3.06 M** | **113** | **4.46 ± 0.44** |
| **ShuffleFAC V1** | | | | | |
| Reference | FP32 | 65.71 ± 2.10 | 1.06 M | 116 | 5.61 ± 0.61 |
| SKD | FP32 | 70.60 ± 0.85 | 1.06 M | 116 | 5.61 ± 0.61 |
| AD | FP32 | 68.81 ± 1.33 | 1.06 M | 116 | 5.61 ± 0.61 |
| QAT | INT8 | 68.79 ± 0.85 | 1.06 M | 74 | 4.16 ± 0.61 |
| **QAD (Ours)** | **INT8** | **70.09 ± 0.29** | **1.06 M** | **74** | **4.16 ± 0.61** |

<p align="left"><sub>$\dagger$ : Measured on Raspberry Pi 5</sub></p>
</div>

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
