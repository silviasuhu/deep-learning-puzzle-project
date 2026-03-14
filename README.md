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

### 3. Dataset Setup: CelebA-HQ

This project uses the CelebAMask-HQ dataset.

#### 3.1 Download the dataset

You can obtain the dataset from either of the following sources:

- [Github repository](https://github.com/switchablenorms/CelebAMask-HQ)
- [Google Drive - downloading link](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view)

Download the CelebAMask-HQ.zip file to your local machine.

#### 3.2 Organize and extract the dataset

Follow these steps to decompress and place the dataset in this repository:

```
cd <root_directory_of_this_repository>
mkdir -p data/CelebA-HQ
mv ~/Downloads/CelebAMask-HQ.zip data/CelebA-HQ/.
unzip data/CelebA-HQ/CelebAMask-HQ.zip -d data/CelebA-HQ
```

After extraction, ensure that the dataset files are correctly placed under

```
data/CelebA-HQ/
```

### 4. Training

To start a training loop, run the script `src/run_training.py` with Python and provide the desired parameters.

The following parameters can be used when calling the script:

```
usage: run_training.py [-h] [-b BATCH_SIZE] [-s STEPS] [-e EPOCHS] [-p PUZZLE_SIZES [PUZZLE_SIZES ...]] [-d DEGREE] [--project_name PROJECT_NAME] [--visual_model VISUAL_MODEL] [--gnn_model GNN_MODEL] [--checkpoint_path CHECKPOINT_PATH] [--missing_percentage MISSING_PERCENTAGE]
                       [--wandb_disabled] [--wandb_project WANDB_PROJECT]

options:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training
  -s STEPS, --steps STEPS
                        Number of diffusion steps
  -e EPOCHS, --epochs EPOCHS
                        Maximum number of epochs to train for
  -p PUZZLE_SIZES [PUZZLE_SIZES ...], --puzzle_sizes PUZZLE_SIZES [PUZZLE_SIZES ...]
                        Input a list of values. They will be used to create puzzles of different sizes during training (for example, if list is 2 4 7 then puzzles will be divided into 2x2, 4x4, and 7x7 pieces).
  -d DEGREE, --degree DEGREE
                        Degree of the expander graph. -1 = fully connected. Default is -1.
  --project_name PROJECT_NAME
                        Project name mainly used for checkpoint naming and Weights & Biases logging if -wandb_project is not set.
  --visual_model VISUAL_MODEL
                        Model used to convert patches to feature embeddings. Options: 'resnet18equiv' or any model accesible by timm. Default is 'resnet18equiv'.
  --gnn_model GNN_MODEL
                        GNN model to use. Options: 'transformer', 'exophormer'. Default is 'transformer'.
  --checkpoint_path CHECKPOINT_PATH
                        Path to the checkpoint to load. If not set, training will start from scratch. If set, the script will attempt to load the checkpoint at the specified path and resume training from that point.
  --missing_percentage MISSING_PERCENTAGE
                        Percentage of missing pieces in the puzzle (0-100). Default is 0 (no missing pieces).
  --wandb_disabled      Disable logging to Weights & Biases. If unset, the script will log to Weights & Biases using the project name specified in -wandb_project or -project_name.
  --wandb_project WANDB_PROJECT
                        Weights & Biases project name. If not set, the script will use the value of -project_name as the project name for Weights & Biases logging.
```

The following command shows an example of how to run a training session using an Exophormer with a 60% graph connectivity, 300 diffusion steps, and images randomly split into 6×6 or 10×10 patches.

```
python src/run_training.py --batch_size 12 --steps 300 --epochs 300 --puzzle_sizes 6 10 --gnn_model 'exophormer' --degree 60
```

Note that if you are not logged into Weight & Biases you'll be prompted to add an API KEY to proceed with the training. If you don't want to use Weight & Biases, you can disable it through the parameter '--wandb_disabled'.

#### Checkpoints

During training:

- the latest checkpoint is always saved to `outputs/checkpoints/<project_name>/last_model.pt`.
- the model that achieves the best position accuracy is saved to `outputs/checkpoints/<project_name>/best_model.py`.
- aditionally, a checkpoint is saved every 5 epochs.

### 5. Evaluation

TODO

#### Inference Animation

TODO

### Notes & Troubleshooting

- Always activate the virtual environment before running any scripts.

- If you encounter installation issues with torch-scatter, double-check your Python version, PyTorch version, and CUDA compatibility.

- For GPU usage, verify CUDA availability with:

```
import torch
print(torch.cuda.is_available())
```

## Experiments

TODO
