import torch
import torch.nn.functional as F


def extract(a, t):
    out = a.gather(-1, t)
    return out[:, None]


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


# Geometry helpers (EXTRA metrics for evaluation)
def predict_x0(x_t, t, pred_noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t)
    return (x_t - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t


def split_pose(x):
    # Assumes node features = [tx, ty, cosθ, sinθ]
    pos = x[:, 0:2]
    rot = x[:, 2:4]
    return pos, rot


def position_error(pred_pos, gt_pos):
    return torch.norm(pred_pos - gt_pos, dim=-1).mean()


def rotation_error(pred_rot, gt_rot):
    pred_rot = F.normalize(pred_rot, dim=-1)
    gt_rot = F.normalize(gt_rot, dim=-1)

    dot = (pred_rot * gt_rot).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(dot)  # radians
    return angle.mean()


def piece_position_accuracy(pred_pos, gt_pos, thresh=0.05):
    dist = torch.norm(pred_pos - gt_pos, dim=-1)
    return (dist < thresh).float().mean()


def piece_rotation_accuracy(pred_rot, gt_rot, rot_thresh_deg=10):
    rot_diff = torch.atan2(
        torch.sin(pred_rot - gt_rot), torch.cos(pred_rot - gt_rot)
    ).abs()

    rot_ok = rot_diff < torch.deg2rad(torch.tensor(rot_thresh_deg))
    return rot_ok.float().mean()


def strict_piece_accuracy(
    pred_pos, gt_pos, pred_rot, gt_rot, pos_thresh=0.05, rot_thresh_deg=10
):

    pos_ok = torch.norm(pred_pos - gt_pos, dim=-1) < pos_thresh

    rot_diff = torch.atan2(
        torch.sin(pred_rot - gt_rot), torch.cos(pred_rot - gt_rot)
    ).abs()

    rot_ok = rot_diff < torch.deg2rad(torch.tensor(rot_thresh_deg))

    return (pos_ok & rot_ok).float().mean()


class GNN_Diffusion:
    def __init__(self, steps):
        self.steps = steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.betas = linear_beta_schedule(timesteps=steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def reverse_step(self, x_t, pred_noise, t, t_scalar):
        a_t = self.alphas[t].unsqueeze(-1)
        ab_t = self.alphas_cumprod[t].unsqueeze(-1)
        b_t = self.betas[t].unsqueeze(-1)

        z = torch.randn_like(x_t) if t_scalar > 0 else torch.zeros_like(x_t)

        # DDPM reverse update: x_t -> x_{t-1}
        return (1.0 / torch.sqrt(a_t)) * (
            x_t - ((1.0 - a_t) / torch.sqrt(1.0 - ab_t + 1e-8)) * pred_noise
        ) + torch.sqrt(b_t) * z

    def sample_pose(self, batch, model):
        # Get num graphs in the batch and start from pure noise.
        num_graphs = int(batch.batch.max().item()) + 1
        x_t = torch.randn_like(batch.x)

        # Compute visual features once and reuse through all reverse steps.
        patch_feats = model.visual_features(batch.patches)

        for t_scalar in reversed(range(self.steps)):
            t_graph = torch.full(
                (num_graphs,),
                t_scalar,
                device=batch.x.device,
                dtype=torch.long,
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

            x_t = self.reverse_step(x_t, pred_noise, t, t_scalar)

        return x_t

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def true_ddim_step(self, x_t, pred_noise, t_current, t_prev):
        # 1. Get the cumulative alphas
        ab_t = self.alphas_cumprod[t_current].unsqueeze(-1)

        if t_prev < 0:
            ab_t_prev = torch.ones_like(ab_t)  # Step 0 has no noise
        else:
            ab_t_prev = self.alphas_cumprod[t_prev].unsqueeze(-1)

        # 2. Predict the perfectly clean state (x_0)
        pred_x0 = (x_t - torch.sqrt(1.0 - ab_t) * pred_noise) / torch.sqrt(ab_t + 1e-8)

        # 3. Calculate the direction pointing to target step
        dir_xt = torch.sqrt(1.0 - ab_t_prev) * pred_noise

        # 4. Combine to get the exact state at t_prev
        x_prev = torch.sqrt(ab_t_prev) * pred_x0 + dir_xt

        return x_prev

    def accelerated_sample_pose(self, batch, model, sampling_steps=50):
        if sampling_steps <= 0:
            raise ValueError("sampling_steps must be a positive integer")

        num_graphs = int(batch.batch.max().item()) + 1
        x_t = torch.randn_like(batch.x)
        patch_feats = model.visual_features(batch.patches)

        sampling_steps = min(int(sampling_steps), self.steps)

        # Create a stable descending sequence of target steps.
        raw_times = torch.linspace(self.steps - 1, 0, steps=sampling_steps)
        times = []
        for t in raw_times.round().to(torch.long).tolist():
            if not times or t != times[-1]:
                times.append(t)

        if times[-1] != 0:
            times.append(0)

        time_pairs = list(zip(times[:-1], times[1:])) + [(times[-1], -1)]

        for t_current_scalar, t_prev_scalar in time_pairs:
            t_graph = torch.full(
                (num_graphs,), t_current_scalar, device=batch.x.device, dtype=torch.long
            )
            t_current = t_graph[batch.batch]

            # Network predicts the noise at the current step
            pred_noise, _ = model.forward_with_feats(
                x_t,
                t_current,
                batch.patches,
                batch.edge_index,
                patch_feats,
                batch.batch,
            )

            # Jump directly to the previous target step
            x_t = self.true_ddim_step(x_t, pred_noise, t_current_scalar, t_prev_scalar)

        return x_t

    def training_step(self, batch, model, criterion, optimizer):

        optimizer.zero_grad()

        # Initialize the variables that will be used
        batch_size = batch.batch.max().item() + 1

        # t is a 1D tensor of size 'batch_size' with random integers between [0 and steps)
        # It represents the diffusion time step for each graph in the batch
        t = torch.randint(0, self.steps, (batch_size,), device=self.device).long()

        # Expand t to match the number of nodes in the batch
        t = torch.gather(t, 0, batch.batch)

        # x_start contains the good positions and rotations of each patch
        x_start = batch.x

        # Get a random noise
        noise = torch.randn_like(x_start)

        # Compute x_noisy

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Compute patch_feats

        patch_feats = model.visual_features(batch.patches)

        # Compute prediction

        prediction, attentions = model.forward_with_feats(
            x_noisy, t, batch.patches, batch.edge_index, patch_feats, batch.batch
        )

        # Compute loss

        target = noise
        loss = criterion(target, prediction)

        loss.backward()
        optimizer.step()
        return loss.item()

    def validation_step(self, batch, model, criterion):

        # move batch to GPU/CPU
        batch = batch.to(self.device)
        batch_size_graphs = batch.batch.max().item() + 1
        # One timestep per graph
        t = torch.randint(
            0, self.steps, (batch_size_graphs,), device=self.device
        ).long()
        # Expand t to node-level: each node gets its graph's timestep
        t = torch.gather(t, 0, batch.batch)
        # clean node feature (positions + rotations, etc.)
        x_start = batch.x
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # CNN features from image patches
        patch_feats = model.visual_features(batch.patches)
        # 6. Predict noise with GNN
        prediction, _ = model.forward_with_feats(
            x_noisy, t, batch.patches, batch.edge_index, patch_feats, batch.batch
        )
        # Compute validation lose
        # Target = true noise
        # Prediction = model's noise estimate
        val_loss = criterion(noise, prediction)
        # Store scalar loss

        # new metrics for evaluation
        # ---- reconstruct predicted clean pose ----
        x0_pred = predict_x0(
            x_noisy,
            t,
            prediction,
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
        )

        gt_pos, gt_rot = split_pose(x_start)
        pred_pos, pred_rot = split_pose(x0_pred)

        pos_err = position_error(pred_pos, gt_pos)
        rot_err = rotation_error(pred_rot, gt_rot)
        acc_pos = piece_position_accuracy(pred_pos, gt_pos)
        acc_rot = piece_rotation_accuracy(pred_rot, gt_rot)
        # acc_strict = strict_piece_accuracy(pred_pos, gt_pos, pred_rot, gt_rot)

        return {
            "loss": val_loss.detach(),
            "pos_error": pos_err.detach(),
            "rot_error": rot_err.detach(),
            "pos_accuracy": acc_pos.detach(),
            "rot_accuracy": acc_rot.detach(),
            # "strict_accuracy": acc_strict.detach(),
        }
