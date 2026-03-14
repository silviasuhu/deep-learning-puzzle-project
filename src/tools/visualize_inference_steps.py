import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset_celeb import CelebA_DataSet
from gnn_diffusion import linear_beta_schedule
from model.efficient_gat import Eff_GAT
from puzzle_dataset import Puzzle_Dataset_ROT


def resolve_checkpoint_path(checkpoint_arg: str) -> Path:
    path = Path(checkpoint_arg)
    if path.exists():
        return path.resolve()

    root_relative = PROJECT_ROOT / checkpoint_arg
    if root_relative.exists():
        return root_relative.resolve()

    outputs_relative = PROJECT_ROOT / "outputs" / "checkpoints" / checkpoint_arg
    if outputs_relative.exists():
        return outputs_relative.resolve()

    matches = sorted(
        (PROJECT_ROOT / "outputs" / "checkpoints").glob(f"**/{checkpoint_arg}")
    )
    if len(matches) == 1:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_arg}. Use absolute path, project-relative path, or checkpoint filename."
    )


def pose_to_xy_theta(x_pose: torch.Tensor):
    xy = x_pose[:, :2].detach().cpu().numpy()
    rot = x_pose[:, 2:4].detach().cpu().numpy()
    theta = np.arctan2(rot[:, 1], rot[:, 0])
    return xy, theta


def rotation_vec_to_quarter_turns(rot_vec: torch.Tensor) -> torch.Tensor:
    angles = torch.atan2(rot_vec[:, 1], rot_vec[:, 0])
    return torch.round(angles / (np.pi / 2.0)).long() % 4


def assign_nodes_to_grid(pos: torch.Tensor, puzzle_size: int) -> list[tuple[int, int]]:
    x_coords = torch.linspace(-1.0, 1.0, puzzle_size, device=pos.device)
    y_coords = torch.linspace(-1.0, 1.0, puzzle_size, device=pos.device)
    grid_xy = torch.stack(torch.meshgrid(x_coords, y_coords, indexing="xy"), dim=-1)
    grid_xy = grid_xy.reshape(-1, 2)

    cost = torch.cdist(pos, grid_xy).detach().cpu().numpy()
    node_ids, cell_ids = linear_sum_assignment(cost)
    return list(zip(node_ids.tolist(), cell_ids.tolist()))


def render_canvas_from_pose(
    patches: torch.Tensor,
    pose: torch.Tensor,
    puzzle_size: int,
) -> torch.Tensor:
    pos = pose[:, :2]
    rot = pose[:, 2:4]
    quarter_turns = rotation_vec_to_quarter_turns(rot)

    _, channels, patch_h, patch_w = patches.shape
    canvas = torch.zeros(
        (channels, puzzle_size * patch_h, puzzle_size * patch_w), dtype=patches.dtype
    )

    assignments = assign_nodes_to_grid(pos, puzzle_size)
    for node_id, cell_id in assignments:
        gx = cell_id % puzzle_size
        gy = cell_id // puzzle_size
        x0 = gx * patch_w
        y0 = gy * patch_h

        piece = patches[node_id]
        piece = torch.rot90(
            piece, k=int((-quarter_turns[node_id]).item()) % 4, dims=(-2, -1)
        )
        canvas[:, y0 : y0 + patch_h, x0 : x0 + patch_w] = piece

    return canvas.clamp(0, 1)


def tensor_image_to_numpy(img_chw: torch.Tensor) -> np.ndarray:
    return img_chw.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)


def save_step_figure(
    patches: torch.Tensor,
    gt_pose: torch.Tensor,
    pred_pose: torch.Tensor,
    puzzle_size: int,
    step_index: int,
    out_path: Path,
    prediction_only: bool,
):
    pred_canvas = render_canvas_from_pose(
        patches=patches, pose=pred_pose, puzzle_size=puzzle_size
    )

    if prediction_only:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(tensor_image_to_numpy(pred_canvas))
        ax.set_title(f"Predicted Reconstruction (step {step_index})")
        ax.axis("off")
    else:
        gt_canvas = render_canvas_from_pose(
            patches=patches, pose=gt_pose, puzzle_size=puzzle_size
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(tensor_image_to_numpy(gt_canvas))
        axes[0].set_title("Ground Truth Target")
        axes[0].axis("off")
        axes[1].imshow(tensor_image_to_numpy(pred_canvas))
        axes[1].set_title(f"Predicted Reconstruction (step {step_index})")
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def should_save_step(step_index: int, total_steps: int, save_every: int) -> bool:
    if step_index == 0 or step_index == total_steps:
        return True
    return step_index % save_every == 0


def maybe_make_gif(sample_dir: Path, fps: int):
    try:
        import imageio.v2 as imageio
    except ImportError:
        print(
            "imageio not installed, skipping GIF generation. Install with: pip install imageio"
        )
        return

    frame_paths = sorted(sample_dir.glob("frame_*.png"))
    if not frame_paths:
        return

    images = [imageio.imread(p) for p in frame_paths]
    gif_path = sample_dir / "animation.gif"
    imageio.mimsave(gif_path, images, duration=1.0 / max(fps, 1))
    print(f"Saved GIF: {gif_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    steps = args.steps
    if steps is None:
        steps = int(checkpoint.get("config", {}).get("steps", 300))

    puzzle_size = args.puzzle_size
    if puzzle_size is None:
        puzzle_cfg = checkpoint.get("config", {}).get("puzzle_sizes", [6])
        puzzle_size = int(puzzle_cfg[0] if isinstance(puzzle_cfg, list) else puzzle_cfg)

    dataset = CelebA_DataSet(args.dataset_path, train=False)
    puzzle_dataset = Puzzle_Dataset_ROT(
        dataset=dataset,
        patch_per_dim=[(puzzle_size, puzzle_size)],
        augment=False,
        degree=args.degree,
        unique_graph=None,
        all_equivariant=False,
        random_dropout=False,
        missing_percentage=args.missing_percentage,
    )
    loader = torch_geometric.loader.DataLoader(
        puzzle_dataset,
        batch_size=args.batch_size,
        shuffle=args.random_sample,
    )

    model = Eff_GAT(
        steps=steps,
        input_channels=4,
        output_channels=4,
        n_layers=4,
        model=args.visual_model,
        architecture=args.gnn_model,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    betas = linear_beta_schedule(timesteps=steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    output_dir = (
        PROJECT_ROOT
        / args.output_dir
        / f"{checkpoint_path.stem}_steps_{steps}_puzzle_{puzzle_size}x{puzzle_size}"
    )
    os.makedirs(output_dir, exist_ok=True)

    visualized = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            if visualized >= args.num_samples:
                break

            batch = batch.to(device)
            num_graphs = int(batch.batch.max().item()) + 1
            patch_feats = model.visual_features(batch.patches)
            x_start = batch.x
            x_t = torch.randn_like(batch.x)

            batch_sample_ids = [int(v) for v in batch.ind_name.view(-1).tolist()]
            selected_local = []
            for local_idx, sample_id in enumerate(batch_sample_ids):
                if visualized + len(selected_local) >= args.num_samples:
                    break
                selected_local.append((local_idx, sample_id))

            for local_idx, sample_id in selected_local:
                sample_dir = output_dir / f"sample_{sample_id:05d}"
                os.makedirs(sample_dir, exist_ok=True)
                node_mask = batch.batch == local_idx
                save_step_figure(
                    patches=batch.patches[node_mask].detach().cpu(),
                    gt_pose=x_start[node_mask].detach().cpu(),
                    pred_pose=x_t[node_mask].detach().cpu(),
                    puzzle_size=puzzle_size,
                    step_index=0,
                    out_path=sample_dir / "frame_0000.png",
                    prediction_only=args.prediction_only,
                )

            for step_index, t_scalar in enumerate(reversed(range(steps)), start=1):
                t_graph = torch.full(
                    (num_graphs,), t_scalar, device=device, dtype=torch.long
                )
                t = t_graph[batch.batch]

                pred_noise, _ = model.forward_with_feats(
                    x_t,
                    t,
                    batch.patches,
                    batch.edge_index,
                    patch_feats,
                    batch.batch,
                )

                a_t = alphas[t].unsqueeze(-1)
                ab_t = alphas_cumprod[t].unsqueeze(-1)
                b_t = betas[t].unsqueeze(-1)
                z = torch.randn_like(x_t) if t_scalar > 0 else torch.zeros_like(x_t)

                x_t = (1.0 / torch.sqrt(a_t)) * (
                    x_t - ((1.0 - a_t) / torch.sqrt(1.0 - ab_t + 1e-8)) * pred_noise
                ) + torch.sqrt(b_t) * z

                if should_save_step(step_index, steps, args.save_every):
                    for local_idx, sample_id in selected_local:
                        sample_dir = output_dir / f"sample_{sample_id:05d}"
                        node_mask = batch.batch == local_idx
                        save_step_figure(
                            patches=batch.patches[node_mask].detach().cpu(),
                            gt_pose=x_start[node_mask].detach().cpu(),
                            pred_pose=x_t[node_mask].detach().cpu(),
                            puzzle_size=puzzle_size,
                            step_index=step_index,
                            out_path=sample_dir / f"frame_{step_index:04d}.png",
                            prediction_only=args.prediction_only,
                        )

            for _, sample_id in selected_local:
                print(f"Saved frames for sample {sample_id}")
                if args.make_gif:
                    maybe_make_gif(output_dir / f"sample_{sample_id:05d}", args.gif_fps)

            visualized += len(selected_local)

    print(f"Done. Visualization files are in: {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run checkpoint inference and save denoising-step visualizations for future GIF creation."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path or checkpoint filename.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "CelebA-HQ"),
        help="Path to CelebA-HQ root directory.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=6, help="Inference batch size."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Diffusion steps. Default: inferred from checkpoint.",
    )
    parser.add_argument(
        "--puzzle_size",
        type=int,
        default=None,
        help="Puzzle size N for NxN. Default: inferred.",
    )
    parser.add_argument(
        "--visual_model",
        type=str,
        default="resnet18equiv",
        help="Visual backbone used in training.",
    )
    parser.add_argument(
        "--gnn_model",
        type=str,
        default="transformer",
        help="GNN architecture used in training.",
    )
    parser.add_argument(
        "--degree", type=int, default=-1, help="Graph degree used in puzzle dataset."
    )
    parser.add_argument(
        "--missing_percentage",
        type=int,
        default=0,
        help="Missing pieces percentage used in dataset setup.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=3, help="Number of samples to visualize."
    )
    parser.add_argument(
        "--save_every", type=int, default=10, help="Save one frame every K steps."
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional batch cap for quick runs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/inference_steps",
        help="Output directory relative to project root.",
    )
    parser.add_argument(
        "--make_gif", action="store_true", help="Generate animation.gif per sample."
    )
    parser.add_argument(
        "--gif_fps", type=int, default=8, help="GIF FPS when --make_gif is set."
    )
    parser.add_argument(
        "--prediction_only",
        action="store_true",
        help="Save frames with only the predicted reconstruction (single panel).",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Randomize sample order so visualized samples are randomly picked from dataset_path.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
