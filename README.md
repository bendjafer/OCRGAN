# OCR-GAN &mdash; Official PyTorch Implementation

Official PyTorch implementation of the paper "[Omni-frequency Channel-selection Representations for Unsupervised Anomaly Detection](https://arxiv.org/pdf/2203.00259.pdf)".

For any inquiries, please contact Yufei Liang at [yufeiliang@zju.edu.cn](mailto:yufeiliang@zju.edu.cn) or [186368@zju.edu.cn](mailto:186368@zju.edu.cn)

## Overview

OCR-GAN (Omni-frequency Channel-selection Representations GAN) is an unsupervised anomaly detection method that leverages frequency domain analysis and channel selection for robust anomaly detection. The model employs a dual-path architecture that processes both low-frequency (Laplacian) and high-frequency (residual) components of input images to capture comprehensive representations for anomaly detection.

### Key Features

- **Dual-path frequency decomposition**: Separates images into Laplacian (low-frequency) and residual (high-frequency) components
- **Generator-Discriminator architecture**: Uses adversarial training for robust feature learning
- **Cross-dataset generalization**: Supports training on merged datasets from multiple sources
- **Comprehensive evaluation**: Tested on MVTec AD, DAGM, and KolektorSDD datasets

## Requirements

This code has been developed under `Python3.7`, `PyTorch 1.2.0` and `CUDA 10.0` on `Ubuntu 16.04`.

```shell
# Install python3 packages
pip install -r requirements.txt
```

## Data Preparation

### Directory Structure

Your data directory should follow this structure:

```
data/
├── merged/          # Cross-dataset merged classes
│   ├── mvtec_merged/
│   ├── dagm_merged/
│   └── kolektorsdd_merged/
├── processed/       # Dataset-specific processed data
│   ├── mvtec_processed/
│   ├── dagm_processed/
│   └── kolektorsdd_processed/
└── unprocessed/     # Raw datasets from sources
    ├── mvtec/
    ├── DAGM/
    └── KolektorSDD/
```

### Dataset Downloads

Download the following datasets and place them in the `data/unprocessed/` directory:

1. **MVTec AD**: [Download from MVTec](www.kaggle.com/code/ipythonx/mvtec-ad-anomaly-detection-with-anomalib-library)
2. **DAGM 2007**: [Download from Kaggle](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection/dagm-2007-competition-dataset-optical-inspection)
3. **KolektorSDD**: [Download from Kaggle](https://www.vicos.si/resources/kolektorsdd/)

### Data Processing Scripts

The `data_creation/` folder contains scripts to process raw datasets into the required format:

#### 1. Dataset-Specific Processing

**`prepare_mvtec.py`**
- Processes MVTec AD dataset into train/test splits
- Organizes defect types into good/bad categories
- Adds class prefixes to image names for identification
- Removes ground truth masks and metadata files

**`prepare_dagm.py`**
- Processes DAGM 2007 dataset (10 classes)
- Separates images based on corresponding label files
- Creates train/good and test/good|bad splits
- Handles Class1-Class10 folder structure

**`prepare_kolektorsdd.py`**
- Processes KolektorSDD dataset (50 sequences)
- Uses label masks to determine good/bad samples
- Balances test set with equal good/bad samples
- Handles kos01-kos50 sequence structure

**`snippets_maker.py`**
- Specialized for video anomaly detection datasets
- Creates frame snippets from video sequences
- Processes UCSD Pedestrian dataset format
- Generates good/bad snippets based on ground truth

#### 2. Cross-Dataset Merging

**`merge_into_single_class.py`**
- Merges multiple dataset classes into single anomaly detection datasets
- Creates cross-class training for improved generalization
- Supports MVTec (15 classes), DAGM (10 classes), and KolektorSDD (50 sequences)

### Processing Commands

To prepare your datasets, execute the following commands in order:

```bash
# Process individual datasets
cd ocrgan_image_adapted/data_creation/
python prepare_mvtec.py
python prepare_dagm.py
python prepare_kolektorsdd.py

# Merge classes for cross-dataset training
python merge_into_single_class.py
```

### Processed Data Structure

After processing, each dataset class follows this structure:

```
data/processed/mvtec_processed/metal_nut/
├── train/
│   └── good/
│       ├── metal_nut_000.png
│       ├── metal_nut_001.png
│       └── ...
└── test/
    ├── good/
    │   ├── metal_nut_000.png
    │   └── ...
    └── bad/
        ├── metal_nut_bent_000.png
        ├── metal_nut_color_001.png
        └── ...
```

## Training

### Training Scripts

The `train/` folder contains various training configurations:

#### Individual Dataset Training

**`train_mvtec.sh`**
- Trains on merged MVTec classes
- Default configuration: 256px images, 200 epochs, batch size 64
- Uses GPU ID 2, OCR-GAN-Aug model

**`train_dagm.sh`**
- Trains on merged DAGM classes  
- Configuration: 256px images, 200 epochs, batch size 64

**`train_kolektorsdd.sh`**
- Trains on merged KolektorSDD sequences
- Similar configuration to other datasets

#### Multi-Class Training

**`train_mvtec_all.sh`**
- Trains individual models for each of the 15 MVTec classes
- Loops through: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper
- Uses `train_each.py` for individual class training

**`train_dagm_all.sh`**
- Trains individual models for each of the 10 DAGM classes
- Processes dagm_1 through dagm_10 classes

#### Master Training Script

**`train.sh`**
- Executes all training scripts in sequence
- Comprehensive training pipeline for all datasets

### Training Commands

```bash
cd ocrgan_image_adapted/

# Train on specific merged dataset
bash train/train_mvtec.sh      # MVTec merged classes
bash train/train_dagm.sh       # DAGM merged classes  
bash train/train_kolektorsdd.sh # KolektorSDD merged sequences

# Train individual class models
bash train/train_mvtec_all.sh  # All 15 MVTec classes separately
bash train/train_dagm_all.sh   # All 10 DAGM classes separately

# Run complete training pipeline
bash train/train.sh            # All training scripts
```

### Training Parameters

Key parameters that can be modified in the training scripts:

- `ISIZE`: Input image size (default: 256)
- `NITER`: Number of training epochs (default: 200)
- `BATCHSIZE`: Training batch size (default: 64)
- `MODEL`: Model architecture (default: "ocr_gan_aug")
- `GPU_ID`: GPU device ID for training
- `DATAROOT`: Path to training data

### Training Monitoring

All training scripts generate:
- Training history logs in `output/history/`
- Model checkpoints in `output/ocr_gan_aug/[dataset]/train/weights/`
- Test images in `output/ocr_gan_aug/[dataset]/train/test_images/`

## Model Architecture

The OCR-GAN model consists of:

1. **Generator (NetG)**: 
   - Dual-path architecture processing Laplacian and residual components
   - Reconstructs both frequency components independently
   - Combines outputs for final reconstruction

2. **Discriminator (NetD)**:
   - Evaluates realism of generated images
   - Provides feature representations for anomaly scoring
   - Uses adversarial loss for training

3. **Frequency Decomposition**:
   - Laplacian component captures low-frequency features
   - Residual component captures high-frequency details
   - Enables comprehensive representation learning

## Evaluation

The model uses multiple evaluation metrics:
- **ROC-AUC**: Area under ROC curve
- **Precision-Recall**: Precision-recall analysis
- **Per-pixel evaluation**: Localization accuracy

### Performance Results

Performance on different datasets:

**KolektorSDD** (after 200 epochs):
- AUC: 0.891
- Max AUC: 0.905

**DAGM** (after 81 epochs):
- AUC: 0.410
- Max AUC: 0.622

**MVTec** (after 26 epochs):
- AUC: 0.461
- Max AUC: 0.582

## Testing

For inference on trained models:

```bash
python test.py --dataset [DATASET_NAME] --isize 256 --model ocr_gan_aug --load_weights
```

## Citation

If our work is helpful for your research, please consider citing:

```bibtex
@article{liang2022omni,
  title={Omni-frequency Channel-selection Representations for Unsupervised Anomaly Detection},
  author={Liang, Yufei and Zhang, Jiangning and Zhao, Shiwei and Wu, Runze and Liu, Yong and Pan, Shuwen},
  journal={arXiv preprint arXiv:2203.00259},
  year={2022}
}
```

## Acknowledgements

We thank the great work [GANomaly](https://github.com/samet-akcay/ganomaly) for providing assistance for our research.
