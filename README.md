# Postgraduate Deep Learning project

This repository contains the code and experiments for a postgraduate-level deep learning project. The instructions below guide you through setting up a reproducible development environment on Linux and preparing the required dataset.

## Prerequisites

- Python 3.12.12 (this exact version was used during development and testing; other versions may work but are not guaranteed)

## Development Environment Setup in Linux

These are the steps to setup a development environment for Linux after cloning this repository:

### 1. Create a Python virtual environment

Note: This repository was developed and tested with Python 3.12.12. Compatibility with earlier or later versions is not guaranteed.

```
cd <root_directory_of_this_repository>
python3 -m venv .venv
```

Activate the virtual environment before running any python scripts of this repository:

```
cd <root_directory_of_this_repository>
source .venv/bin/activate
```

### 2. Install Python Dependencies

To avoid compatibility issues between PyTorch and PyTorch Geometric, install the packages in the order shown below.

#### 2.1. Install PyTorch, TorchVision, and Torch Geometric

Choose the section that matches your hardware setup.

##### Option A: GPU (CUDA 12.6)

```
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.6.0+cu126.html
pip install torch-geometric
```

##### Option B: CPU-only

```
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.6.0+cpu.html
pip install torch-geometric
```

Note: Mixing CUDA and CPU wheels will lead to runtime errors. Ensure that your PyTorch and torch-scatter installations target the same backend.

#### 2.2 Install remaining requirements

Install all other Python dependencies listed in requirements.txt:

```
pip install -r requirements.txt
```

## 3. Dataset Setup: CelebA-HQ

This project uses the CelebAMask-HQ dataset.

### 3.1 Download the dataset

You can obtain the dataset from either of the following sources:

- [Github repository](https://github.com/switchablenorms/CelebAMask-HQ)
- [Google Drive - downloading link](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view)

Download the CelebAMask-HQ.zip file to your local machine.

### 3.2 Organize and extract the dataset

Follow these steps to decompress and place the dataset in this repository:

```
cd <root_directory_of_this_repository>
mkdir -p data/CelebA-HQ/images
mv ~/Downloads/CelebAMask-HQ.zip data/CelebA-HQ/images/.
unzip data/CelebA-HQ/images/CelebAMask-HQ.zip -d data/CelebA-HQ/images
```

After extraction, ensure that the dataset files are correctly placed under

```
data/CelebA-HQ/
```

### 4. Next steps [WIP]

- Training and evaluation instructions
- Configuration options
- Experiment reproducibility notes
  These sections will be added as the project evolves.

## Notes & Troubleshooting

- Always activate the virtual environment before running any scripts.

- If you encounter installation issues with torch-scatter, double-check your Python version, PyTorch version, and CUDA compatibility.

- For GPU usage, verify CUDA availability with:
