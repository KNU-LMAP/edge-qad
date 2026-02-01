# Quantization Adversarial Distillation for Embedded Underwater Acoustic Target Recognition
QAD, In preparation for **IEEE Geoscience and Remote Sensing Letters(GRSL) 2026**

## Introduction

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
2. Go to data/ & Run split_data.py(Set your Raw file root and preprocessing output root)
```bash
cd data/
python split_data.py
```

## Train
1. Go to scripts/
```bash
cd ..
cd scripts/
```
2. Open *.sh file and Set your the preprocessing output root
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
