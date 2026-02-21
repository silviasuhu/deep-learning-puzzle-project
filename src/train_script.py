import argparse
from pathlib import Path
import torch
import torch_geometric
import numpy as np

from dataset_celeb_rot import CelebA_DataSet, CelebA_Graph_Dataset
from puzzle_dataset import Puzzle_Dataset_ROT
from gnn_diffusion import GNN_Diffusion
from transformers.optimization import Adafactor
from model.efficient_gat import Eff_GAT
from torch.utils.data import random_split
import wandb

from datetime import datetime


def main(
    batch_size: int, steps: int, epochs: int, puzzle_sizes: list, wandb_disabled: bool
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Eff_GAT(
        steps=steps,
        input_channels=4,
        output_channels=4,
        n_layers=4,
        model="resnet18equiv",
    )
    model.to(device)

    criterion = torch.nn.functional.smooth_l1_loss
    optimizer = Adafactor(model.parameters())

    # Create a list of tuples containing the actual puzzle sizes
    # If puzzles_sizes is [2, 4, 7], then patch_per_dim will be [(2, 2), (4, 4), (7, 7)]
    patch_per_dim = [(x, x) for x in puzzle_sizes]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Start a new wandb run to track this script.
    run = (
        wandb.init(
            entity="postgraduate-project-puzzle-upc",
            project="Puzzle Diffusion_GNN",
            name=f"{timestamp}_Puzzle{puzzle_sizes}_steps{steps}_bs{batch_size}",
            # Track hyperparameters and run metadata.
            config={
                "batch_size": batch_size,
                "steps": steps,
                "epochs": epochs,
                "patch_per_dim": patch_per_dim,
                "model": "Eff_gat",
                "optimizer": "Adafactor",
                "loss": "smooth_l1",
            },
        )
        if not wandb_disabled
        else wandb.init(mode="offline")
    )

    train_dt = CelebA_DataSet(train=True)
    train_validation_dataset = Puzzle_Dataset_ROT(
        dataset=train_dt,
        patch_per_dim=patch_per_dim,
        augment=False,
        degree=-1,
        unique_graph=None,
        all_equivariant=False,
        random_dropout=False,
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

    gnn_diffusion = GNN_Diffusion(steps=steps)

    checkpoint_dir = Path("outputs") / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):

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
        val_acc = []

        # disable gradient tracking (save memory,prevents accidental backprop)
        with torch.no_grad():
            for batch in val_loader:
                val_metrics = gnn_diffusion.validation_step(batch, model, criterion)
                val_loss.append(val_metrics["loss"].item())
                val_pos.append(val_metrics["pos_error"].item())
                val_rot.append(val_metrics["rot_error"].item())
                val_acc.append(val_metrics["accuracy"].item())

        val_loss_mean = np.mean(val_loss)
        val_pos_mean = np.mean(val_pos)
        val_rot_mean = np.mean(val_rot)
        val_acc_mean = np.mean(val_acc)

        # -------- LOGGING --------
        run.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss_mean,
                "val/pos_error": val_pos_mean,
                "val/rot_error_rad": val_rot_mean,
                "val/rot_error_deg": val_rot_mean * 180.0 / np.pi,
                "val/accuracy": val_acc_mean,
            }
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss_mean:.4f} | "
            f"Pos Err: {val_pos_mean:.4f} | "
            f"Rot Err: {val_rot_mean * 180.0 / np.pi:.2f} deg | "
            f"Accuracy: {val_acc_mean:.4f}"
        )

        #   ---- CHECKPOINT ----
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            checkpoint_path = checkpoint_dir / f"model_epoch{epoch+1}.pt"

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
                    "val_accuracy": val_acc_mean,
                },
            }

            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-batch_size", type=int, default=6)
    ap.add_argument("-steps", type=int, default=300)
    ap.add_argument("-epochs", type=int, default=1)
    ap.add_argument("-wandb_disabled", action="store_true")
    ap.add_argument(
        "-puzzle_sizes",
        nargs="+",
        default=[6],
        type=int,
        help="Input a list of values. They will be used to create puzzles of different sizes during training (for example, if list is 2 4 7 then puzzles will be divided into 2x2, 4x4, and 7x7).",
    )

    args = ap.parse_args()
    print(args)

    main(
        batch_size=args.batch_size,
        steps=args.steps,
        epochs=args.epochs,
        puzzle_sizes=args.puzzle_sizes,
        wandb_disabled=args.wandb_disabled,
    )
