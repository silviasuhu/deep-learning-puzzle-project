import argparse
from pathlib import Path
import torch
import torch_geometric
import numpy as np
import logging

from dataset_celeb_rot import CelebA_DataSet, CelebA_Graph_Dataset
from puzzle_dataset import Puzzle_Dataset_ROT
from gnn_diffusion import GNN_Diffusion
from transformers.optimization import Adafactor
from model.efficient_gat import Eff_GAT
from torch.utils.data import random_split
import wandb

# You can adjust the logging level to:
#    - DEBUG to see all messages
#    - INFO to see only higher-level messages
#    - WARNING to see only warnings and errors
#    - ERROR to see only errors
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


def main(batch_size: int, steps: int, epochs: int, puzzle_sizes: list):

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

    # Start a new wandb run to track this script.
    run = wandb.init(
        entity="postgraduate-project-puzzle-upc",
        project="my-awesome-project",
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
        val_losses = []

        # disable gradient tracking (save memory,prevents accidental backprop)
        with torch.no_grad():
            for batch in val_loader:
                val_loss = gnn_diffusion.validation_step(batch, model, criterion)
                val_losses.append(val_loss.item())

        val_loss_mean = np.mean(val_losses)
        # -------- LOGGING --------
        run.log(
            {"epoch": epoch + 1, "train/loss": train_loss, "val/loss": val_loss_mean}
        )

        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss_mean:.4f}"
        )
        run.log({"train/loss": train_loss, "epoch": epoch + 1})
        run.finish()

        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss_mean:.4f}"
        )

        # ---- CHECKPOINT ----
        # Save a checkpoint every 5 epochs and at the last epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            checkpoint_path = checkpoint_dir / f"model_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-batch_size", type=int, default=6)
    ap.add_argument("-steps", type=int, default=300)
    ap.add_argument("-epochs", type=int, default=1)
    ap.add_argument(
        "-puzzle_sizes",
        nargs="+",
        default=[6],
        type=int,
        help="Input a list of values. They will be used to create puzzles of different sizes during training (for example, if list is 2 4 7 then puzzles will be divided into 2x2, 4x4, and 7x7).",
    )

    args = ap.parse_args()
    logger.info(f"Arguments: {args}")

    main(args.batch_size, args.steps, args.epochs, args.puzzle_sizes)
