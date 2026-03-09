import sys
import argparse
from pathlib import Path
import torch
import torch_geometric
import numpy as np

from dataset_celeb import CelebA_DataSet
from puzzle_edgefeat_dataset import Puzzle_Dataset_Edge_ROT
from edgegnn_diffusion import EdgeGNN_Diffusion
from transformers.optimization import Adafactor
from model.border_gat import Border_GAT
from torch.utils.data import random_split
import wandb

from datetime import datetime


def main(
    batch_size: int,
    steps: int,
    epochs: int,
    puzzle_sizes: list,
    wandb_disabled: bool,
    checkpoint_load_path: str,
    wandb_project: str,
    project_name: str,
    visual_model: str,
    gnn_model: str,
    degree: int,
    missing_percentage: int,
):
    print(f"Cuda is available: {torch.cuda.is_available()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_acc_pos = 0.0

    if not wandb_project:
        wandb_project = project_name

    epoch_offset = 0
    if checkpoint_load_path:
        checkpoint = torch.load(checkpoint_load_path, weights_only=False)

        epoch_offset = checkpoint["epoch"]
        steps = checkpoint["config"]["steps"]
        puzzle_sizes = checkpoint["config"]["puzzle_sizes"]

        model = Border_GAT(
            steps=steps,
            input_channels=4,
            output_channels=4,
            n_layers=4,
            model=visual_model,
            architecture=gnn_model,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer = Adafactor(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        best_acc_pos = (
            checkpoint["metrics"]["best_pos_accuracy"]
            or checkpoint["metrics"]["val_pos_accuracy"]
        )

        print(
            f"Loaded checkpoint from {checkpoint_load_path} at epoch {epoch_offset} with best position accuracy {best_acc_pos:.4f}"
        )

    else:
        model = Border_GAT(
            steps=steps,
            input_channels=4,
            output_channels=4,
            n_layers=4,
            model=visual_model,
            architecture=gnn_model,
            #freeze_backbone=True
        )
        optimizer = Adafactor(model.parameters())

    checkpoint_dir = Path("outputs") / "checkpoints" / project_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # If last_model.pt exists, notify user and stop, to prevent overwriting
    if not checkpoint_load_path and (checkpoint_dir / "last_model.pt").exists():
        print(
            f"Warning: {checkpoint_dir / 'last_model.pt'} already exists. To prevent overwriting, the script will stop. If you want to continue training from this checkpoint, use the --checkpoint_load_path argument with the path to this file. If you want to start a new training run, you can delete the file, move it to a different location or use a different '-project_name' parameter."
        )
        sys.exit(1)

    model.to(device)

    criterion = torch.nn.functional.smooth_l1_loss

    # Create a list of tuples containing the actual puzzle sizes
    # If puzzles_sizes is [2, 4, 7], then patch_per_dim will be [(2, 2), (4, 4), (7, 7)]
    patch_per_dim = [(x, x) for x in puzzle_sizes]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Start a new wandb run to track this script.
    run = (
        wandb.init(
            entity="postgraduate-project-puzzle-upc",
            project=wandb_project,
            # project="my-awesome-project",
            name=f"{timestamp}_Puzzle{puzzle_sizes}_steps{steps}_bs{batch_size}",
            # Track hyperparameters and run metadata.
            config={
                "batch_size": batch_size,
                "steps": steps,
                "epochs": epochs,
                "epoch_offset": epoch_offset,
                "puzzle_sizes": puzzle_sizes,
                "model": "Border_GAT",
                "optimizer": "Adafactor",
                "loss": "smooth_l1",
                "checkpoint_load_path": checkpoint_load_path,
                "edge_features": "border_similarity",
                "edge_embedding_dim": 32
            },
        )
        if not wandb_disabled
        else wandb.init(mode="offline")
    )

    train_dt = CelebA_DataSet(train=True)
    train_validation_dataset = Puzzle_Dataset_Edge_ROT(
        dataset=train_dt,
        patch_per_dim=patch_per_dim,
        augment=False,
        degree=degree,
        unique_graph=None,
        all_equivariant=False,
        random_dropout=False,
        #missing_percentage=missing_percentage,
    )
    # split dataset training and validation:
    val_ratio = 0.1
    val_size = int(len(train_validation_dataset) * val_ratio)
    train_size = len(train_validation_dataset) - val_size

    train_dataset, val_dataset = random_split(
        train_validation_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = torch_geometric.loader.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch_geometric.loader.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    gnn_diffusion = EdgeGNN_Diffusion(steps=steps)

    for e in range(epochs):
        epoch = e + epoch_offset

        # -------- TRAIN --------
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = batch.to(device)
            loss = gnn_diffusion.training_step(batch, model, criterion, optimizer)
            # losses.append(loss)
            train_losses.append(loss)
        train_loss = np.mean(train_losses)

        # VALIDATION
        # switch model to evaluation mode
        model.eval()
        val_loss = []
        val_pos = []
        val_rot = []
        val_acc_pos = []
        val_rot_acc = []
        # val_strict_acc = []

        # disable gradient tracking (save memory,prevents accidental backprop)
        with torch.no_grad():
            for batch in val_loader:
                val_metrics = gnn_diffusion.validation_step(batch, model, criterion)
                val_loss.append(val_metrics["loss"].item())
                val_pos.append(val_metrics["pos_error"].item())
                val_rot.append(val_metrics["rot_error"].item())
                val_acc_pos.append(val_metrics["pos_accuracy"].item())
                val_rot_acc.append(val_metrics["rot_accuracy"].item())
                # val_strict_acc.append(val_metrics["strict_accuracy"].item())

        val_loss_mean = np.mean(val_loss)
        val_pos_mean = np.mean(val_pos)
        val_rot_mean = np.mean(val_rot)
        val_acc_pos_mean = np.mean(val_acc_pos)
        val_rot_acc_mean = np.mean(val_rot_acc)
        # val_strict_acc_mean = np.mean(val_strict_acc)

        # -------- LOGGING --------
        run.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss_mean,
                "val/pos_error": val_pos_mean,
                "val/rot_error_rad": val_rot_mean,
                "val/rot_error_deg": val_rot_mean * 180.0 / np.pi,
                "val/pos_accuracy": val_acc_pos_mean,
                "val/rot_accuracy": val_rot_acc_mean,
                # "val/strict_accuracy": val_strict_acc_mean,
            }
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss_mean:.4f} | "
            f"Pos Err: {val_pos_mean:.4f} | "
            f"Rot Err: {val_rot_mean * 180.0 / np.pi:.2f} deg | "
            f"Pos Accuracy: {val_acc_pos_mean:.4f} | "
            f"Rot Accuracy: {val_rot_acc_mean:.4f} | "
        )
        #   ---- CHECKPOINT ----
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "steps": steps,
                "batch_size": batch_size,
                "puzzle_sizes": puzzle_sizes,
            },
            "metrics": {
                "val_loss": val_loss_mean,
                "val_pos_error": val_pos_mean,
                "val_rot_error": val_rot_mean,
                "val_pos_accuracy": val_acc_pos_mean,
                "val_rot_accuracy": val_rot_acc_mean,
                "best_pos_accuracy": best_acc_pos,
            },
        }

        # Store the best checkpoint
        if val_acc_pos_mean > best_acc_pos:
            best_acc_pos = val_acc_pos_mean
            checkpoint_save_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_save_path)
            print(f"Saved checkpoint: {checkpoint_save_path}")

        # Store checkpoint every 5 epochs and in the last epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            checkpoint_save_path = checkpoint_dir / f"model_epoch{epoch+1}.pt"
            torch.save(checkpoint, checkpoint_save_path)
            print(f"Saved checkpoint: {checkpoint_save_path}")

        # Always store last checkpoint
        checkpoint_save_path = checkpoint_dir / "last_model.pt"
        torch.save(checkpoint, checkpoint_save_path)
        print(f"Saved checkpoint: {checkpoint_save_path}")

    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument(
        "-b", "--batch_size", type=int, default=6, help="Batch size for training"
    )
    ap.add_argument(
        "-s", "--steps", type=int, default=300, help="Number of diffusion steps"
    )
    ap.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1,
        help="Maximum number of epochs to train for",
    )
    ap.add_argument(
        "-p",
        "--puzzle_sizes",
        nargs="+",
        default=[6],
        type=int,
        help="Input a list of values. They will be used to create puzzles of different sizes during training (for example, if list is 2 4 7 then puzzles will be divided into 2x2, 4x4, and 7x7 pieces).",
    )
    ap.add_argument(
        "-d",
        "--degree",
        type=int,
        default=-1,
        help="Degree of the expander graph. -1 = fully connected. Default is -1.",
    )
    ap.add_argument(
        "--project_name",
        type=str,
        default="puzzle",
        help="Project name mainly used for checkpoint naming and Weights & Biases logging if -wandb_project is not set.",
    )
    ap.add_argument(
        "--visual_model",
        type=str,
        default="resnet18equiv",
        help="Model used to convert patches to feature embeddings. Options: 'resnet18equiv' or any model accesible by timm. Default is 'resnet18equiv'.",
    )
    ap.add_argument(
        "--gnn_model",
        type=str,
        default="transformer",
        help="GNN model to use. Options: 'transformer', 'exophormer'. Default is 'transformer'.",
    )
    ap.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to the checkpoint to load. If not set, training will start from scratch. If set, the script will attempt to load the checkpoint at the specified path and resume training from that point.",
    )
    ap.add_argument(
        "--missing_percentage",
        type=int,
        default=0,
        help="Percentage of missing pieces in the puzzle (0-100). Default is 0 (no missing pieces).",
    )
    ap.add_argument(
        "--wandb_disabled",
        action="store_true",
        help="Disable logging to Weights & Biases. If unset, the script will log to Weights & Biases using the project name specified in -wandb_project or -project_name.",
    )
    ap.add_argument(
        "--wandb_project",
        type=str,
        help="Weights & Biases project name. If not set, the script will use the value of -project_name as the project name for Weights & Biases logging.",
    )

    args = ap.parse_args()
    print(args)

    main(
        batch_size=args.batch_size,
        steps=args.steps,
        epochs=args.epochs,
        puzzle_sizes=args.puzzle_sizes,
        wandb_disabled=args.wandb_disabled,
        wandb_project=args.wandb_project,
        project_name=args.project_name,
        visual_model=args.visual_model,
        gnn_model=args.gnn_model,
        degree=args.degree,
        checkpoint_load_path=args.checkpoint_path,
        missing_percentage=args.missing_percentage,
    )
