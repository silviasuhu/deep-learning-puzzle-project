import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric

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


def infer_puzzle_size(checkpoint: dict, explicit_size: int | None) -> int:
    if explicit_size is not None:
        return explicit_size

    puzzle_cfg = checkpoint.get("config", {}).get("puzzle_sizes", [6])
    if isinstance(puzzle_cfg, list):
        return int(puzzle_cfg[0])
    return int(puzzle_cfg)


def select_timesteps(
    total_steps: int, requested: list[int] | None, count: int
) -> list[int]:
    if requested:
        valid = sorted(
            {int(t) for t in requested if 0 <= int(t) < total_steps}, reverse=True
        )
        if not valid:
            raise ValueError(
                f"No valid timesteps found. Use values in [0, {total_steps - 1}] for this checkpoint."
            )
        return valid

    count = max(1, min(count, total_steps))
    raw = torch.linspace(total_steps - 1, 0, steps=count)
    return sorted({int(t.item()) for t in raw.round().to(torch.long)}, reverse=True)


def wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def deterministic_reverse_mean(
    x_t: torch.Tensor,
    pred_noise: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:
    return (1.0 / torch.sqrt(alpha_t)) * (
        x_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t + 1e-8)) * pred_noise
    )


def save_vector_field_plot(
    sample_dir: Path,
    sample_id: int,
    step_number: int,
    t_scalar: int,
    x_t_sample: torch.Tensor,
    x_mean_prev_sample: torch.Tensor,
    x_gt_sample: torch.Tensor,
):
    current_pos = x_t_sample[:, :2].detach().cpu().numpy()
    mean_prev_pos = x_mean_prev_sample[:, :2].detach().cpu().numpy()
    gt_pos = x_gt_sample[:, :2].detach().cpu().numpy()

    translation_delta = mean_prev_pos - current_pos
    translation_magnitude = np.linalg.norm(translation_delta, axis=-1)

    current_angle = np.arctan2(
        x_t_sample[:, 3].detach().cpu().numpy(),
        x_t_sample[:, 2].detach().cpu().numpy(),
    )
    mean_prev_angle = np.arctan2(
        x_mean_prev_sample[:, 3].detach().cpu().numpy(),
        x_mean_prev_sample[:, 2].detach().cpu().numpy(),
    )
    rotation_delta = wrap_angle_rad(mean_prev_angle - current_angle)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    axes[0].scatter(
        gt_pos[:, 0], gt_pos[:, 1], c="lightgray", s=55, label="ground truth"
    )
    quiver = axes[0].quiver(
        current_pos[:, 0],
        current_pos[:, 1],
        translation_delta[:, 0],
        translation_delta[:, 1],
        translation_magnitude,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        cmap="viridis",
        width=0.006,
    )
    axes[0].scatter(
        current_pos[:, 0], current_pos[:, 1], c="black", s=18, label="current x_t"
    )
    axes[0].set_title(f"Translation Drift at t={t_scalar}")
    axes[0].legend(loc="upper right")
    fig.colorbar(quiver, ax=axes[0], fraction=0.046, pad=0.04, label="|mean drift|")

    rot_scatter = axes[1].scatter(
        current_pos[:, 0],
        current_pos[:, 1],
        c=np.degrees(rotation_delta),
        cmap="coolwarm",
        vmin=-180.0,
        vmax=180.0,
        s=80,
    )
    axes[1].scatter(gt_pos[:, 0], gt_pos[:, 1], c="lightgray", s=28)
    axes[1].set_title("Rotation Correction (degrees)")
    fig.colorbar(rot_scatter, ax=axes[1], fraction=0.046, pad=0.04, label="delta theta")

    axes[2].scatter(
        gt_pos[:, 0], gt_pos[:, 1], c="tab:green", s=55, label="ground truth"
    )
    axes[2].scatter(
        current_pos[:, 0], current_pos[:, 1], c="tab:orange", s=35, label="current x_t"
    )
    axes[2].scatter(
        mean_prev_pos[:, 0],
        mean_prev_pos[:, 1],
        c="tab:blue",
        s=28,
        label="mean x_{t-1}",
    )
    for node_idx in range(current_pos.shape[0]):
        axes[2].plot(
            [current_pos[node_idx, 0], mean_prev_pos[node_idx, 0]],
            [current_pos[node_idx, 1], mean_prev_pos[node_idx, 1]],
            color="tab:blue",
            alpha=0.5,
            linewidth=1.0,
        )
    axes[2].set_title("Current vs Mean Reverse Step")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(
        f"Checkpoint Vector Field | sample {sample_id:05d} | reverse step {step_number} | diffusion t={t_scalar}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(sample_dir / f"vector_field_t{t_scalar:04d}.png", dpi=150)
    plt.close(fig)


def maybe_make_gif(sample_dir: Path, fps: int):
    try:
        import imageio.v2 as imageio
    except ImportError:
        print(
            "imageio not installed, skipping GIF generation. Install with: pip install imageio"
        )
        return

    frame_paths = sorted(sample_dir.glob("vector_field_t*.png"), reverse=True)
    if not frame_paths:
        return

    images = [imageio.imread(p) for p in frame_paths]
    gif_path = sample_dir / "vector_field_animation.gif"
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

    puzzle_size = infer_puzzle_size(checkpoint, args.puzzle_size)
    selected_timesteps = set(
        select_timesteps(steps, args.timesteps, args.num_snapshots)
    )

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
        / f"{checkpoint_path.stem}_vector_fields_steps_{steps}_puzzle_{puzzle_size}x{puzzle_size}"
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
                sample_dir = output_dir / f"sample_{sample_id:05d}"
                os.makedirs(sample_dir, exist_ok=True)

            for step_number, t_scalar in enumerate(reversed(range(steps)), start=1):
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

                alpha_t = alphas[t].unsqueeze(-1)
                alpha_bar_t = alphas_cumprod[t].unsqueeze(-1)
                mean_x_prev = deterministic_reverse_mean(
                    x_t=x_t,
                    pred_noise=pred_noise,
                    alpha_t=alpha_t,
                    alpha_bar_t=alpha_bar_t,
                )

                if t_scalar in selected_timesteps:
                    for local_idx, sample_id in selected_local:
                        node_mask = batch.batch == local_idx
                        sample_dir = output_dir / f"sample_{sample_id:05d}"
                        save_vector_field_plot(
                            sample_dir=sample_dir,
                            sample_id=sample_id,
                            step_number=step_number,
                            t_scalar=t_scalar,
                            x_t_sample=x_t[node_mask],
                            x_mean_prev_sample=mean_x_prev[node_mask],
                            x_gt_sample=x_start[node_mask],
                        )

                z = torch.randn_like(x_t) if t_scalar > 0 else torch.zeros_like(x_t)
                beta_t = betas[t].unsqueeze(-1)
                x_t = mean_x_prev + torch.sqrt(beta_t) * z

            for _, sample_id in selected_local:
                if args.make_gif:
                    maybe_make_gif(output_dir / f"sample_{sample_id:05d}", args.gif_fps)

            visualized += len(selected_local)

    print(f"Saved vector field plots to: {output_dir}")
    print(
        "Timesteps visualized: "
        + ", ".join(str(t) for t in sorted(selected_timesteps, reverse=True))
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained checkpoint and visualize the learned reverse-diffusion field "
            "on puzzle poses. Each saved figure shows translation drift, rotation correction, "
            "and the deterministic mean reverse step at selected timesteps."
        )
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
        "--batch_size",
        type=int,
        default=4,
        help="Inference batch size.",
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
        help="Puzzle size N for NxN. Default: inferred from checkpoint.",
    )
    parser.add_argument(
        "--visual_model",
        type=str,
        default="resnet18equiv",
        help="Visual backbone used to train the checkpoint.",
    )
    parser.add_argument(
        "--gnn_model",
        type=str,
        default="transformer",
        help="GNN architecture used to train the checkpoint.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=-1,
        help="Graph degree used in puzzle dataset.",
    )
    parser.add_argument(
        "--missing_percentage",
        type=int,
        default=0,
        help="Missing pieces percentage used in dataset setup.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="Number of samples to visualize.",
    )
    parser.add_argument(
        "--timesteps",
        nargs="+",
        type=int,
        default=None,
        help="Exact diffusion timesteps to visualize, for example: --timesteps 299 200 100 0.",
    )
    parser.add_argument(
        "--num_snapshots",
        type=int,
        default=6,
        help="Number of evenly spaced timesteps to save when --timesteps is not provided.",
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
        default="outputs/vector_fields",
        help="Output directory relative to project root.",
    )
    parser.add_argument(
        "--make_gif",
        action="store_true",
        help="Generate vector_field_animation.gif per sample.",
    )
    parser.add_argument(
        "--gif_fps",
        type=int,
        default=4,
        help="GIF FPS when --make_gif is set.",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Randomize sample order so visualized samples are randomly picked from the dataset.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
