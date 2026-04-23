MBAN: Multi-branch Attention Network for Liver Fibrosis Staging in Multi-phase MRI

## Method

The proposed MBAN framework consists of three main components:

### 1. Multi-branch Encoder
Multiple modality-specific branches are used to independently extract fibrosis-related intra-sequence features from multi-phase MRI.

### 2. Adaptive Branch Squeeze-and-Excitation Module (ABSM)
A branch-wise attention fusion module is used to dynamically recalibrate the importance of different MRI branches and reduce feature redundancy.

### 3. Lightweight Attention Bottleneck (LAB)
A lightweight Transformer-based bottleneck is used to model both inter-branch dependencies and intra-sequence structural information.

Finally, an MLP-based classifier outputs the predictions for liver fibrosis staging tasks.

## Data

The method is designed for multi-phase liver MRI, including:

- T1WI
- T2WI
- DWI
- Arterial phase
- Venous phase
- Delayed phase
- Hepatobiliary phase

Following the paper, T1WI and T2WI are used as shared anatomical inputs, and they are paired with DWI or dynamic contrast-enhanced phases to form five modality-specific inputs.

## Preprocessing

All MRI sequences are preprocessed as follows:

- spatial registration to a common anatomical space
- resampling each volume to 32 slices
- cropping and resizing each slice to 192 × 192
- per-volume normalization to zero mean and unit variance
- random rotation, scaling, and flipping during training

The final input shape for each modality is:

```text
[32, 192, 192]

## Tasks

This repository supports two binary classification tasks:

- **Cirrhosis detection**: S1-S3 vs. S4
- **Substantial fibrosis detection**: S1 vs. S2-S4

The model is trained in a multi-task learning manner using the sum of two binary classification losses.

## Repository Structure

```text
MBAN/
├── configs/
│   └── mban.yaml
├── datasets/
│   ├── lifs_dataset.py
│   └── transforms.py
├── models/
│   ├── mban.py
│   ├── encoder.py
│   ├── absm.py
│   ├── lab.py
│   └── classifier.py
├── tools/
│   ├── train.py
│   ├── validate.py
│   ├── test.py
│   └── infer.py
├── utils/
│   ├── losses.py
│   ├── metrics.py
│   └── misc.py
├── requirements.txt
└── README.md

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/MBAN.git
cd MBAN

## TODO

- [ ] Release training code
- [ ] Release inference code
- [ ] Release preprocessing scripts
- [ ] Release pretrained checkpoints
- [ ] Add visualization examples
- [ ] Add dataset preparation instructions
