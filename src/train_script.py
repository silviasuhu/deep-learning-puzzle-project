import argparse
import torch
import torch_geometric
import numpy as np

from dataset_celeb_rot import CelebA_DataSet, CelebA_Graph_Dataset
from puzzle_dataset import Puzzle_Dataset_ROT
from gnn_diffusion import GNN_Diffusion
from transformers.optimization import Adafactor
from model.efficient_gat import Eff_GAT


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

    train_dt = CelebA_DataSet(train=True)
    dataset = Puzzle_Dataset_ROT(
        dataset=train_dt,
        patch_per_dim=patch_per_dim,
        augment=False,
        degree=-1,
        unique_graph=None,
        all_equivariant=False,
        random_dropout=False,
    )

    dataloader = torch_geometric.loader.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    gnn_diffusion = GNN_Diffusion(steps=steps)

    model.train()
    for epoch in range(epochs):
        losses = []
        for batch in dataloader:
            loss = gnn_diffusion.training_step(batch, model, criterion, optimizer)
            losses.append(loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {np.mean(losses):.4f}")


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
    print(args)
    main(args.batch_size, args.steps, args.epochs, args.puzzle_sizes)
