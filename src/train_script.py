import argparse
import torch
import numpy as np

from dataset import CelebA_DataSet, CelebA_Graph_Dataset
from model.efficient_gat import Eff_GAT
from puzzle_dataset import Puzzle_Dataset


def training_step(model, batch, criterion, optimizer, steps):
    optimizer.zero_grad()

    batch_size = batch.batch.max().item() + 1

    # t is a 1D tensor of size 'batch_size' with random integers between [0 and steps)
    # It represents the diffusion time step for each graph in the batch
    t = torch.randint(0, steps, (batch_size,), device=model.device).long()

    # Expand t to match the number of nodes in the batch
    new_t = torch.gather(t, 0, batch.batch)

    # TODO

    # for key in batch.keys():
    #     print("SSS batch key:", key)

    outputs = model(batch)
    loss = criterion(outputs, batch.y)
    loss.backward()
    optimizer.step()
    return loss.item()


def main(batch_size, steps, epochs, puzzle_sizes):
    model = Eff_GAT(steps=steps)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # dataset = CelebA_Graph_Dataset(
    #     dataset=CelebA_DataSet(
    #         images_path="data/CelebA-HQ/images/CelebAMask-HQ/CelebA-HQ-img/",
    #         txt_path="data/CelebA-HQ/CelebA-HQ_train.txt",
    #     ),
    #     num_patches_x=patch_size_x,
    #     num_patches_y=patch_size_y,
    # )

    dataset = Puzzle_Dataset(
        dataset=CelebA_DataSet(
            images_path="data/CelebA-HQ/images/CelebAMask-HQ/CelebA-HQ-img/",
            txt_path="data/CelebA-HQ/CelebA-HQ_train.txt",
        ),
        patch_per_dim=[(x, x) for x in puzzle_sizes],
        augment=False,
        degree=-1,
        unique_graph=None,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    model.train()
    for epoch in range(epochs):
        losses = []
        for batch in dataloader:
            loss = training_step(model, batch, criterion, optimizer, steps)
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
