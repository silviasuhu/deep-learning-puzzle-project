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

To evaluate a trained checkpoint on the test split, run the script `src/run_evaluation.py` and provide the corresponding parameters.

The following parameters can be used when calling the script:

```
usage: run_evaluation.py [-h] [-dataset_path DATASET_PATH] [-batch_size BATCH_SIZE] [-steps STEPS] [-model_checkpoint MODEL_CHECKPOINT] [-puzzle_sizes PUZZLE_SIZES] [-visual_model VISUAL_MODEL] [-gnn_model GNN_MODEL] [-degree DEGREE]

options:
  -h, --help            show this help message and exit
  -dataset_path DATASET_PATH
                        Path to the root of the CelebA-HQ dataset directory. Expected to contain 'images/' and split text files. Default: data/CelebA-HQ
  -batch_size BATCH_SIZE
                        Number of puzzle graphs to process per batch during inference. Default: 6
  -steps STEPS          Number of DDPM reverse-diffusion steps to run. Should match the value used when training the checkpoint. Default: 300
  -model_checkpoint MODEL_CHECKPOINT
                        Filename of the checkpoint to load from outputs/checkpoints/ (e.g. 'last_model.pt' or 'best_model.pt').
  -puzzle_sizes PUZZLE_SIZES
                        Grid size of the puzzle to evaluate (single integer). E.g. 6 produces a 6×6 puzzle. Only one size can be tested at a time. Default: 6
  -visual_model VISUAL_MODEL
                        Backbone used to embed image patches into feature vectors. Use 'resnet18equiv' for the equivariant ResNet-18, or any model name supported by timm. Must match training.
  -gnn_model GNN_MODEL
                        GNN architecture used to predict noise over the pose graph. Options: 'transformer', 'exophormer', "edge_transformer". Must match training.
  -degree DEGREE        Degree of the expander graph that defines patch connectivity. -1 means fully connected. Must match training. Default: -1
```

The following command shows an example of how to evaluate a transformer checkpoint on 6×6 puzzles with 300 diffusion steps.

```
python src/run_evaluation.py -model_checkpoint last_model.pt -puzzle_sizes 6 -steps 300 -gnn_model exophormer -visual_model resnet18equiv -degree 60
```

Evaluation results are saved under:

```
test_outputs/<checkpoint_stem>_puzzle_<N>x<N>_inference_results.txt
```

The output report includes setup details and aggregate metrics over all test batches (mean and standard deviation):

- Average position error
- Standard deviation of position error
- Average rotation error
- Standard deviation of rotation error
- Average position accuracy
- Standard deviation of position accuracy
- Average rotation accuracy
- Standard deviation of rotation accuracy

#### 6. Inference Animation

A notebook is provided in "notebooks/inference_visualization_shared_pipeline.ipynb" to perform a denoising of a single sample and visualize the result. 

Contains the same parameters as the `src/run_evaluation.py`, with additional options for visualization customization:

```
additional options:

  -save_every N          Save frame from denoising trajectory every n steps.
  -random_sample T/F     Shuffle samples when creating the DataLoader.          
  -random_seed int/None  Use a random seed to ensure sample picking reproducibility. Use "None" to pick a random sample every time.
  -prediction_only T/F   Run only denoising, without generating visualization.       
  -make_gif T/F          Make an animated .gif with all the sampled frames.
  -gif_fps int           FPS of the gif.
  -max_batches  int      Number of batches to process.
  -output_dir DIR        Directory to save the metrics and visualizations.
  -sampler  DDPM/DDIM    Sample to use for denoising. DDIM is faster and allows lower number of steps than training. DDPM must have the same number of steps as training.
  -sampling_steps int    Number of denoising stpes. Only used if sampler == "DDIM"
  -verbose_progress T/F  Print progress. 
```

### Notes & Troubleshooting

- Always activate the virtual environment before running any scripts.

- If you encounter installation issues with torch-scatter, double-check your Python version, PyTorch version, and CUDA compatibility.

- For GPU usage, verify CUDA availability with:

```
import torch
print(torch.cuda.is_available())
```

## Experiments

The current implementation has been trained varying the GNN model implementation, the number of pieces in the puzzle training set, and the percentage of missing pieces in the puzzles:

| Diffusion steps | Batch size | Model           | Sizes   | Degree | Epochs | Missing pieces | GPU                      | GPU-RAM |
|-----------------|------------|-----------------|---------|--------|-------|----------------|--------------------------|---------|
|             300 |         10 | Transformer     |       6 |    100 |   150 |              0 | RTX 3060Ti               | 8GB     |
|             300 |         64 | Expohormer      |       6 |     60 |   150 |              0 | RTX H100                 | 80GB    |
|             300 |         10 | Exophormer      | 6--7--8 |     60 |   500 |              0 | RTX A4000                | 16GB    |
|             300 |         10 | Expohormer      | 6 to 20 |     60 |   400 |              0 | RTX H100                 | 80GB    |
|             300 |         12 | Transformer     |       6 |    100 |   135 |            10% | NVIDIA T4 (Google Cloud) | 16GB    |
|             300 |        100 | Transformer     |       6 |    100 |   135 |            20% | RTX H100                 | 80GB    |
|             300 |        100 | EdgeTransformer |       6 |    100 |   200 |             0% | RTX H100                 | 80GB    |

Each model was then evaluated using DDPM, 300 denoising steps, for puzzles of different number of pieces. Several metrics were collected:

- Position error (for each image, mean Euclidean distance between the predicted patch position and the ground truth):

<p align="center">
<img width="140" height="57" alt="pos_err" src="https://github.com/user-attachments/assets/047d0fad-f576-48cc-b798-ad62096e82f3" />
</p>

- Position accuracy (fraction of pieces whose prediction falls within a distance threshold (0.05 r.d.u) from the ground truth):
- 
<p align="center">
<img width="203" height="57" alt="pos_acc" src="https://github.com/user-attachments/assets/1d44f90e-e9c7-4830-9272-e8a0cfd79114" />
</p>

- Rotation error (for each image, mean angular distance between the predicted patch position and the ground truth):

<p align="center">
<img width="311" height="57" alt="rot_err" src="https://github.com/user-attachments/assets/3127df01-ea26-4fc9-b9c3-3533a70853a8" />
</p>

- Rotation accuracy (fraction of pieces whose prediction falls within an angle threshold (10 rad) from the ground truth):

<p align="center">
<img width="461" height="57" alt="rot_acc" src="https://github.com/user-attachments/assets/913a037a-09a1-45bf-937b-ba881734d887" />
</p>


### Mean position accuracy results of the whole test dataset:

| Model           | Trained Sizes         | Degree | Epoch | Missing pieces | Accuracy 4x4 | Accuracy 6x6 | Accuracy 7x7 | Accuracy 8x8 | Accuracy 10x10 | Accuracy 14x14 | Accuracy 15x15 | Accuracy 19x19 | Accuracy 20x20 |
|-----------------|-----------------------|--------|-------|----------------|--------------|--------------|--------------|--------------|----------------|----------------|----------------|----------------|----------------|
| Transformer     |                     6 |    100 |   150 |              0 |       0,0348 |       0,9375 |       0,0054 |       0,0124 |         0,0123 |         0,0049 |          0,005 |        -       |        -       |
| Expohormer      |                     6 |     60 |   150 |              0 |       0,0204 |       0,9623 |       0,0012 |       0,0028 |         0,0036 |         0,0027 |         0,0021 |         0,0017 |         0,0021 |
| Exophormer      |           6 -- 7 -- 8 |     60 |   500 |              0 |       0,0153 |       0,9885 |        0,989 |       0,8203 |         0,0385 |         0,0129 |         0,0072 |         0,0042 |         0,0046 |
| Expohormer      | 6 8 10 12 14 16 18 20 |     60 |   400 |              0 |        0,067 |       0,9636 |       0,0085 |       0,7256 |         0,9373 |         0,9294 |          0,012 |          0,004 |         0,7192 |
| Transformer     |                     6 |    100 |   135 |            10% |       0,0183 |       0,9024 |       0,0068 |       0,0155 |          0,009 |         0,0038 |         0,0048 |        -       |        -       |
| Transformer     |                     6 |    100 |   200 |            20% |       0,0224 |       0,6976 |       0,0084 |       0,0092 |         0,0078 |         0,0041 |         0,0042 |        -       |        -       |
| EdgeTransformer |                     6 |    100 |   150 |             0% |       0,0431 |       0,8068 |       0,0125 |       0,0254 |          0,013 |         0,0052 |        -       |        -       |        -       |

### Mean rotation accuracy results of the whole test dataset:

| Model           | Trained Sizes         | Degree | Epoch | Missing pieces | Accuracy 4x4 | Accuracy 6x6 | Accuracy 7x7 | Accuracy 8x8 | Accuracy 10x10 | Accuracy 14x14 | Accuracy 15x15 | Accuracy 19x19 | Accuracy 20x20 |
|-----------------|-----------------------|--------|-------|----------------|--------------|--------------|--------------|--------------|----------------|----------------|----------------|----------------|----------------|
| Transformer     |                     6 |    100 |   150 |              0 |       0,7773 |        0,963 |       0,6037 |       0,6273 |          0,563 |         0,5006 |         0,4768 |        -       |        -       |
| Expohormer      |                     6 |     60 |   150 |              0 |       0,6791 |       0,9746 |       0,4301 |       0,5599 |         0,4852 |         0,4325 |         0,3572 |         0,3504 |         0,3956 |
| Exophormer      |           6 -- 7 -- 8 |     60 |   500 |              0 |       0,7691 |       0,9853 |       0,9895 |       0,9035 |         0,7111 |         0,4774 |         0,5107 |         0,4785 |         0,4703 |
| Expohormer      | 6 8 10 12 14 16 18 20 |     60 |   400 |              0 |       0,7902 |       0,9731 |       0,5588 |       0,8875 |          0,953 |         0,9513 |         0,7342 |          0,772 |         0,9499 |
| Transformer     |                     6 |    100 |   135 |            10% |       0,7709 |       0,9486 |       0,5812 |       0,6182 |         0,5509 |         0,4796 |         0,4656 |        -       |        -       |
| Transformer     |                     6 |    100 |   200 |            20% |       0,8122 |       0,9118 |       0,5847 |       0,6158 |         0,5209 |           0,46 |         0,4418 |        -       |        -       |
| EdgeTransformer |                     6 |    100 |   150 |             0% |       0,8514 |       0,9589 |       0,6186 |       0,6725 |         0,5648 |         0,4942 |        -       |        -       |        -       |
