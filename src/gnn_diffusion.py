import torch


def extract(a, t):
    out = a.gather(-1, t)
    return out[:, None]


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class GNN_Diffusion:
    def __init__(self, steps):
        self.steps = steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # calculations for diffusion q(x_t | x_{t-1}) and others
        betas = linear_beta_schedule(timesteps=steps)
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
        print("Start training_step")

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
        print("Compute x_noisy")
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Compute patch_feats
        print("Compute patch_feats")
        patch_feats = model.visual_features(batch.patches)

        # Compute prediction
        print("Compute prediction")
        prediction, attentions = model.forward_with_feats(
            x_noisy, t, batch.patches, batch.edge_index, patch_feats, batch.batch
        )

        # Compute loss
        print("Compute loss")
        target = noise
        loss = criterion(target, prediction)

        loss.backward()
        optimizer.step()
        return loss.item()
