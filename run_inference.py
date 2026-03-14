# %% [markdown]
# # Run inference
#
# This runs an inference/forward step on a non-training, non-validation dataset and returns:
#
#  - Individual accuracy for translation and rotation for each batch
#  - Global accuracy for translation and rotation the whole set
#
#  Requires:
#
#  - Test dataset path
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

from src.model.full_models import *
from src.puzzle_dataset import *
from src.gnn_diffusion import *
from src.dataset_celeb import CelebA_DataSet


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
        missing_percentage=0,
    )

    # %% [markdown]
    # ## 2.- Load model with checkpoint

    # %%
    # Load checkpoint first so we can infer steps from it
    checkpoint = torch.load(
        f"outputs/checkpoints/{model_checkpoint}",
        weights_only=False,
        map_location=device,
    )

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    # Infer steps from the checkpoint's time embedding weight shape
    if "time_emb.weight" in state_dict:
        steps = state_dict["time_emb.weight"].shape[0]
        print(f"Inferred steps from checkpoint: {steps}", flush=True)

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

    model.load_state_dict(state_dict)
    model.eval()

    print("Model parameters after loading checkpoint:", flush=True)
    for name, param in model.named_parameters():
        print(name, param, flush=True)

    # %% [markdown]
    # ## 3.- Run inference for the whole dataset

    # %% [markdown]
    #

    # %%
    # Dataloader for inference
    test_loader = torch_geometric.loader.DataLoader(
        test_puzzle_dt, batch_size=batch_size, shuffle=False
    )

    # %%
    # Function to add rotational info for the model to work
    def add_rot(batch):
        # Add/force "no rotation" feature [1, 0] for every node in the batch
        N = batch.x.size(0)  # total nodes across all graphs in batch
        rot = torch.zeros(N, 2, dtype=batch.x.dtype, device=batch.x.device)
        rot[:, 0] = 1.0

        if batch.x.size(1) == 2:
            batch.x = torch.cat([batch.x, rot], dim=1)  # [N,4]
        else:
            batch.x[:, 2:4] = rot  # overwrite existing rot channels

        batch.rot = rot
        batch.rot_index = torch.zeros(N, dtype=torch.long, device=batch.x.device)

        return batch

    # %%
    # Switch model to evaluation mode
    model.eval()

    # Prepare diffusion model
    gnn_diffusion = GNN_Diffusion(steps=steps)

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

            # Add rotation dimensions and send batch to device
            batch = add_rot(batch).to(device)

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
                    device=gnn_diffusion.device,
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
    os.makedirs("test_outputs", exist_ok=True)

    # %%
    with open(
        f"test_outputs/{model_checkpoint.split('.')[0]}_puzzle_{puzzle_sizes}x{puzzle_sizes}_inference_results.txt",
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
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument(
        "-dataset_path", type=str, default=os.path.join(os.getcwd(), "data/CelebA-HQ")
    )
    ap.add_argument("-batch_size", type=int, default=6)
    ap.add_argument("-steps", type=int, default=300)
    ap.add_argument("-model_checkpoint", type=str, default="model_epoch10.pt")
    ap.add_argument(
        "-puzzle_sizes",
        default=6,
        type=int,
        help="Puzzle size for the inference (only one size can be tested at a time, so not a list).",
    )
    ap.add_argument("-visual_model", type=str, default="resnet18equiv")
    ap.add_argument("-gnn_model", type=str, default="transformer")
    ap.add_argument(
        "-degree",
        type=int,
        default=-1,
        help="Degree of the expander graph. -1 = fully connected",
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
    )
