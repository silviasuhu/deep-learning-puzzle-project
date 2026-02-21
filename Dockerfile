# This Dockerfile sets up an environment for training a deep learning model using PyTorch.
# It uses a base image that includes CUDA and cuDNN for GPU support, and installs the necessary dependencies
# from the requirements.txt file. The entry point is set to run the training script located in the src folder.
#
# ------------------------------------------ REQUIREMENTS -------------------------------------------------
# - This docker image is designed to be built and run on a machine with an NVIDIA GPU and the
#       appropriate drivers installed.
#
# - You should be logged to Wandb in the terminal before running the container, as the training
#       script uses Wandb for logging (unless you disable it).
#
# - The dataset and checkpoints directories should be mounted as volumes when running the container,
#       so that the training script can access the data and save checkpoints. Therefore, you'll need
#       to download the dataset before running the container. See README.md for instructions on how
#       to download the dataset.
# 
# ----------------------------------- BUILDING AND RUNNING THE CONTAINER -----------------------------------
#
# To build the image, use the following command in the terminal:
#   `cd <directory_containing_this_Dockerfile>`
#   `docker build -t deep-learning-puzzle-project .`
#
# To run the container, use the following command (adjusting the parameters as needed). Note that 
# we use a wandb wrapper to run the container, which allows us to easily log the training process to
# Wandb.
#   `cd <path_to_project_root>`
#   `wandb docker-run --rm -v ./checkpoints:/app/checkpoints -v ./data:/app/data deep-learning-puzzle-project -batch_size 16 -steps 100 -epochs 1000 -puzzle_sizes 6 8 12 ...`

FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy src folder
COPY src src

ENTRYPOINT ["python3", "src/train_script.py"]