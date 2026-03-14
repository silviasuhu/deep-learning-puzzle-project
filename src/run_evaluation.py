# %% [markdown]
# # Run evaluation
#
# This runs an evaluation on the test dataset and returns:
#  - Individual accuracy for translation and rotation for each batch
#  - Global accuracy for translation and rotation the whole set
#
#  Requires:
#  - Model checkpoint

# %% [markdown]
# ## 0.- Load modules + Set paths and variables

# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from model.efficient_gat import Eff_GAT
from puzzle_dataset import *
from dataset_celeb import CelebA_DataSet
from gnn_diffusion import *


# %%
def main(
    dataset_path: str,
    batch_size: int,
    steps: int,
    puzzle_sizes: list,
    model_checkpoint: str,
    visual_model: str,
    gnn_model: str,
    degree: int,
    missing_percentage: int,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Set up:\n{'--'*len(dataset_path)}", flush=True)
    print(f"Dataset path: {dataset_path}", flush=True)
    print(f"Model checkpoint: {model_checkpoint}", flush=True)
    print(f"Visual model: {visual_model}", flush=True)
    print(f"Attention GNN model: {gnn_model}", flush=True)
    print(f"Diffusion steps to run: {steps}", flush=True)
    print(f"Puzzle size: {puzzle_sizes}x{puzzle_sizes}", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    print(f"Using device: {device}", flush=True)

    # %% [markdown]
    # ## 1.- Load test dataset

    # %%
    # Load base dataset
    test_dataset_base = CelebA_DataSet(dataset_path, train=False)

    # Create puzzle dataset
    # Load puzzle dataset and sample an element
    test_puzzle_dt = Puzzle_Dataset_ROT(
        dataset=test_dataset_base,
        patch_per_dim=[(puzzle_sizes, puzzle_sizes)],
        augment=False,
        degree=degree,
        unique_graph=None,
        all_equivariant=False,
        random_dropout=False,
        missing_percentage=missing_percentage,
    )

    # %% [markdown]
    # ## 2.- Load model with checkpoint

    # %%
    # Load checkpoint first so we can infer steps from it
    checkpoint = torch.load(
        PROJECT_ROOT / "outputs" / "checkpoints" / model_checkpoint,
        weights_only=False,
        map_location=device,
    )

    # Load model
    model = Eff_GAT(
        steps=steps,
        input_channels=4,
        output_channels=4,
        n_layers=4,
        model=visual_model,
        architecture=gnn_model,
    )

    # Send model to device
    model.to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model parameters after loading checkpoint:", flush=True)
    for name, param in model.named_parameters():
        print(name, param, flush=True)

    # %% [markdown]
    # ## 3.- Run inference for the whole dataset

    # %%
    # Dataloader for inference
    test_loader = torch_geometric.loader.DataLoader(
        test_puzzle_dt, batch_size=batch_size, shuffle=False
    )

    # %%
    # Switch model to evaluation mode
    model.eval()

    # same schedule as in GNN_Diffusion
    # calculations for diffusion q(x_t | x_{t-1}) and others
    betas = linear_beta_schedule(timesteps=steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # Initialize lists to store metrics for each batch
    test_pos = []
    test_rot = []
    test_acc_pos = []
    test_rot_acc = []

    # Disable gradient tracking (save memory,prevents accidental backprop)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Print info
            print(
                f"Processing batch num {i}/{len(test_loader)} with batch_size {batch.batch.max().item()+1}",
                flush=True,
            )

            batch = batch.to(device)

            # Get num batches (graphs) in the current batch of data
            num_graphs = int(batch.batch.max().item()) + 1

            # CNN features from image patches
            patch_feats = model.visual_features(batch.patches)

            # Initial, clean pose
            x_start = batch.x

            # Start from pure noise
            x_t = torch.randn_like(batch.x)

            # Run step-wise inference
            for t_scalar in reversed(range(steps)):

                t_graph = torch.full(
                    (num_graphs,),
                    t_scalar,
                    device=device,
                    dtype=torch.long,
                )
                t = t_graph[batch.batch]  # node-level timestep

                pred_noise, _ = model.forward_with_feats(
                    x_t, t, batch.patches, batch.edge_index, patch_feats, batch.batch
                )

                # Extract scalars for current timestep t
                a_t = alphas[t].unsqueeze(-1)
                ab_t = alphas_cumprod[t].unsqueeze(-1)
                b_t = betas[t].unsqueeze(-1)

                z = torch.randn_like(x_t) if t_scalar > 0 else torch.zeros_like(x_t)

                # DDPM reverse step: x_t -> x_{t-1}
                x_t = (1.0 / torch.sqrt(a_t)) * (
                    x_t - ((1.0 - a_t) / torch.sqrt(1.0 - ab_t + 1e-8)) * pred_noise
                ) + torch.sqrt(b_t) * z

            # At the end of the diffusion process, x_t should be the predicted clean pose
            x_0 = x_t
            # Get position and rotation of the predicted and ground truth poses
            gt_pos, gt_rot = split_pose(x_start)
            pred_pos, pred_rot = split_pose(x_0)

            # Compute metrics
            pos_err = position_error(pred_pos, gt_pos)
            rot_err = rotation_error(pred_rot, gt_rot)
            acc_pos = piece_position_accuracy(pred_pos, gt_pos)
            acc_rot = piece_rotation_accuracy(pred_rot, gt_rot)

            # Append metrics
            test_pos.append(pos_err.cpu().numpy())
            test_rot.append(rot_err.cpu().numpy())
            test_acc_pos.append(acc_pos.cpu().numpy())
            test_rot_acc.append(acc_rot.cpu().numpy())

            print(
                f"Batch {i} - Position error: {pos_err:.4f}, \
                    Rotation error: {rot_err:.4f}, \
                    Position accuracy: {acc_pos:.4f}, \
                    Rotation accuracy: {acc_rot:.4f}",
                flush=True,
            )

        # Average metrics
        test_pos_mean = np.mean(test_pos)
        test_rot_mean = np.mean(test_rot)
        test_acc_pos_mean = np.mean(test_acc_pos)
        test_rot_acc_mean = np.mean(test_rot_acc)

        test_pos_std = np.std(test_pos)
        test_rot_std = np.std(test_rot)
        test_acc_pos_std = np.std(test_acc_pos)
        test_rot_acc_std = np.std(test_rot_acc)

    # %% [markdown]
    # ## 4.- Save the data

    # %%
    output_dir = PROJECT_ROOT / "test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # %%
    with open(
        output_dir
        / f"{Path(model_checkpoint).stem}_puzzle_{puzzle_sizes}x{puzzle_sizes}_inference_results.txt",
        "w",
    ) as f:
        f.write(f"Set up:\n{'--'*len(dataset_path)}\n")
        f.write(f"Dataset path: {dataset_path}\n")
        f.write(f"Number of test samples: {len(test_puzzle_dt)}\n")
        f.write(f"Number of batches: {len(test_loader)}\n")
        f.write(f"Model checkpoint: {model_checkpoint}\n")
        f.write(f"Visual model: {visual_model}\n")
        f.write(f"Attention GNN model: {gnn_model}\n")
        f.write(f"Diffusion steps to run: {steps}\n")
        f.write(f"Puzzle size: {puzzle_sizes}x{puzzle_sizes}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Using device: {device}\n")

        f.write(f"\n\nInference results:\n{'--'*len(dataset_path)}\n")
        f.write(f"Average position error: {test_pos_mean:.4f}\n")
        f.write(f"Standard deviation of position error: {test_pos_std:.4f}\n")
        f.write(f"Average rotation error: {test_rot_mean:.4f}\n")
        f.write(f"Standard deviation of rotation error: {test_rot_std:.4f}\n")
        f.write(f"Average position accuracy: {test_acc_pos_mean:.4f}\n")
        f.write(f"Standard deviation of position accuracy: {test_acc_pos_std:.4f}\n")
        f.write(f"Average rotation accuracy: {test_rot_acc_mean:.4f}\n")
        f.write(f"Standard deviation of rotation accuracy: {test_rot_acc_std:.4f}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate a trained puzzle-solving model on the CelebA-HQ test split. "
            "Runs the full DDPM reverse diffusion process for each batch and reports "
            "position/rotation error and accuracy (mean ± std across batches). "
            "Results are saved to test_outputs/<checkpoint>_puzzle_NxN_inference_results.txt."
        )
    )

    # Add the arguments to the parser
    ap.add_argument(
        "-dataset_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "CelebA-HQ"),
        help=(
            "Path to the root of the CelebA-HQ dataset directory. "
            "Expected to contain 'images/' and split text files. "
            f"Default: {PROJECT_ROOT / 'data' / 'CelebA-HQ'}"
        ),
    )
    ap.add_argument(
        "-batch_size",
        type=int,
        default=6,
        help="Number of puzzle graphs to process per batch during inference. Default: 6",
    )
    ap.add_argument(
        "-steps",
        type=int,
        default=300,
        help=(
            "Number of DDPM reverse-diffusion steps to run. "
            "Should match the value used when training the checkpoint. Default: 300"
        ),
    )
    ap.add_argument(
        "-model_checkpoint",
        type=str,
        default="",
        help=(
            "Filename of the checkpoint to load from outputs/checkpoints/ "
            "(e.g. 'last_model.pt' or 'best_model.pt'). Required."
        ),
    )
    ap.add_argument(
        "-puzzle_sizes",
        default=6,
        type=int,
        help=(
            "Grid size of the puzzle to evaluate (single integer). "
            "E.g. 6 produces a 6×6 puzzle. Only one size can be tested at a time. Default: 6"
        ),
    )
    ap.add_argument(
        "-visual_model",
        type=str,
        default="resnet18equiv",
        help=(
            "Backbone used to embed image patches into feature vectors. "
            "Use 'resnet18equiv' for the equivariant ResNet-18, or any model name "
            "supported by the timm library. Must match the architecture used during training. "
            "Default: 'resnet18equiv'"
        ),
    )
    ap.add_argument(
        "-gnn_model",
        type=str,
        default="transformer",
        help=(
            "GNN architecture used to predict noise over the pose graph. "
            "Options: 'transformer', 'exophormer'. Must match the architecture used during training. "
            "Default: 'transformer'"
        ),
    )
    ap.add_argument(
        "-degree",
        type=int,
        default=-1,
        help=(
            "Degree of the expander graph that defines patch connectivity. "
            "-1 means fully connected (all patches see each other). "
            "Must match the value used during training. Default: -1"
        ),
    )
    ap.add_argument(
        "-missing_percentage",
        type=int,
        default=0,
        help="Percentage of missing pieces in the puzzle (0-100). Default is 0 (no missing pieces).",
    )

    args = ap.parse_args()
    print(args)

    main(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        steps=args.steps,
        puzzle_sizes=args.puzzle_sizes,
        visual_model=args.visual_model,
        gnn_model=args.gnn_model,
        degree=args.degree,
        model_checkpoint=args.model_checkpoint,
        missing_percentage=args.missing_percentage,
    )
