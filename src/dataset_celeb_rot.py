import os
import torch
import torch.nn.functional as F
import einops
from PIL import Image
import numpy as np
import torch_geometric
from torch_geometric.data import Data
import torchvision.transforms.functional as TF


def split_image_into_patches(image, num_patches_x, num_patches_y):
    """
    Split a CxHxW image (non-overlapping patches) and normalizes coordinates to [-1, 1]
    Centered at 0,0 -> centered,symmetric,numerically stable for diffusion models).
    """

    channels, height, width = image.shape
    # Calculate patch size
    patch_h = height // num_patches_y
    patch_w = width // num_patches_x

    # Resize image to fit exact number of patches
    h_crop = patch_h * num_patches_y
    w_crop = patch_w * num_patches_x
    image = image[:, :h_crop, :w_crop]

    unfolded = F.unfold(
        image.unsqueeze(0),
        kernel_size=(patch_h, patch_w),
        stride=(patch_h, patch_w),
    )

    patches = unfolded.view(
        channels, patch_h, patch_w, num_patches_y, num_patches_x
    ).permute(
        3, 4, 0, 1, 2
    )  # [h, w, C, ph, pw]

    # Create normalized coordinates [-1, 1]
    y_coords = torch.linspace(-1, 1, num_patches_y)
    x_coords = torch.linspace(-1, 1, num_patches_x)

    xy = torch.stack(torch.meshgrid(x_coords, y_coords, indexing="xy"), -1)
    xy = xy.permute(1, 0, 2)  # [grid_h, grid_w, 2]

    return xy, patches


def fully_connected_edge_index(num_nodes, device):
    """
    Create a fully-connected (dense) edge list without self-loops.
    """
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    row = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
    col = torch.arange(num_nodes, device=device).repeat(num_nodes)
    mask = row != col
    edge_index = torch.stack([row[mask], col[mask]], dim=0)
    return edge_index


class CelebA_DataSet(torch.utils.data.Dataset):
    """
    ONLY loads images.
    No patches. No graphs. No diffusion logic.
    """

    def __init__(self, path="data/CelebA-HQ", train=True, transform=None):
        self.images_path = f"{path}/CelebAMask-HQ/CelebA-HQ-img/"
        if train:
            txt_path = f"{path}/CelebA-HQ_train.txt"
        else:
            txt_path = f"{path}/CelebA-HQ_test.txt"

        self.image_names = []
        with open(txt_path, "r", encoding="utf-8") as f:
            self.image_names = f.read().splitlines()

        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.images_path, self.image_names[idx]))
        if self.transform:
            img = self.transform(img)
        return img


class CelebA_Graph_Dataset(torch_geometric.data.Dataset):
    def __init__(
        self,
        dataset=None,
        num_patches_x=6,
        num_patches_y=6,
        drop_ratio=0.0,
        image_size=None,
    ):
        super().__init__()

        self.dataset = dataset
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        self.drop_ratio = drop_ratio
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def get(self, idx):
        image = self.dataset[idx]

        if self.image_size is not None:
            # Resize to fixed resolution for batching
            # Crop afterwards to ensure exact patch grid (inside split_image_into_patches)
            image = TF.resize(image, [self.image_size, self.image_size])

        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)

        if not isinstance(image, torch.Tensor):
            raise TypeError("Dataset must return a torch.Tensor or PIL.Image.")

        # grid-aware split
        coordinates_xy, patches = split_image_into_patches(
            image, self.num_patches_x, self.num_patches_y
        )

        # Flatten patches and coordinates
        patches = einops.rearrange(patches, "h w c ph pw -> (h w) c ph pw")
        coordinates_xy = einops.rearrange(coordinates_xy, "h w c -> (h w) c")

        num_patches = patches.size(0)

        # --------------------------------------------------
        # DiffAssemble-style discrete per-patch rotation
        # --------------------------------------------------
        random_rot = torch.randint(0, 4, (num_patches,))
        rot_one_hot = torch.nn.functional.one_hot(random_rot, 4).float()

        # rotation group representation (matches their code)
        rots = torch.tensor(
            [[1, 0], [0, 1], [-1, 0], [0, -1]],
            dtype=torch.float,
        )
        rots_tensor = rot_one_hot @ rots  # [N, 2]

        # Rotate patch images ONLY (coordinates remain canonical)
        patches_np = (patches * 255).byte().permute(0, 2, 3, 1).cpu().numpy()

        patches_img = [Image.fromarray(p) for p in patches_np]
        rotated_patches = [
            p.rotate(int(r.item()) * 90) for p, r in zip(patches_img, random_rot)
        ]

        patches = torch.stack([TF.to_tensor(p) for p in rotated_patches])

        # Concatenate rotation target to node features (as in Puzzle_Dataset_ROT)
        x = torch.cat([coordinates_xy, rots_tensor], dim=1)

        # Shuffle patches and node features together
        perm = torch.randperm(num_patches)
        patches = patches[perm]
        x = x[perm]
        rots_tensor = rots_tensor[perm]
        random_rot = random_rot[perm]

        # Optionally drop some patches
        if self.drop_ratio > 0:
            keep = int((1 - self.drop_ratio) * patches.size(0))
            patches = patches[:keep]
            x = x[:keep]
            rots_tensor = rots_tensor[:keep]
            random_rot = random_rot[:keep]

        edge_index = fully_connected_edge_index(patches.size(0), patches.device)

        indexes = torch.arange(patches.size(0))

        # Convert to graph data (DiffAssemble-compatible)
        data = Data(
            x=x,
            patches=patches,
            rot=rots_tensor,
            rot_index=random_rot,
            indexes=indexes,
            edge_index=edge_index,
            patches_dim=torch.tensor([self.num_patches_y, self.num_patches_x]),
            ind_name=torch.tensor([idx]),
        )

        return data


if __name__ == "__main__":
    dataset = CelebA_DataSet(train=True)  # or False for test set

    print(f"Dataset size: {len(dataset)}")

    sample_img = dataset[0]
    if hasattr(sample_img, "size"):
        print(f"Sample image size (PIL): {sample_img.size}")
    else:
        print(f"Sample image tensor shape: {sample_img.shape}")

    graph_dataset = CelebA_Graph_Dataset(
        dataset,
        num_patches_x=6,
        num_patches_y=6,
        image_size=192,  # optional but recommended
    )

    graph = graph_dataset.get(0)

    print("\n--- Graph info ---")
    print(f"Graph nodes: {graph.num_nodes}")
    print(f"Graph edges: {graph.edge_index.size(1)}")

    print("\n--- Node features ---")
    print(f"x shape: {graph.x.shape}")  # [N, 4] = (x,y) + rotation embedding
    print(f"patches shape: {graph.patches.shape}")  # [N, C, H, W]

    print("\n--- Rotation supervision ---")
    print(f"rot shape: {graph.rot.shape}")  # [N, 2]
    print(f"rot_index shape: {graph.rot_index.shape}")  # [N]

    print("\n--- Meta ---")
    print(f"indexes shape: {graph.indexes.shape}")
    print(f"patches_dim: {graph.patches_dim}")
