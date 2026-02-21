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


def piece_accuracy(pred_pos, gt_pos, thresh=0.05):
    dist = torch.norm(pred_pos - gt_pos, dim=-1)
    return (dist < thresh).float().mean()


class GNN_Diffusion:
    def __init__(self, steps):
        self.steps = steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # calculations for diffusion q(x_t | x_{t-1}) and others
        betas = linear_beta_schedule(timesteps=steps).to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

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
        acc = piece_accuracy(pred_pos, gt_pos)

        return {
            "loss": val_loss.detach(),
            "pos_error": pos_err.detach(),
            "rot_error": rot_err.detach(),
            "accuracy": acc.detach(),
        }
