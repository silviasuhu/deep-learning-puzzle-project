import os
import torch
import torch.nn.functional as F
import einops
from torch.utils.data import Dataset
from PIL import Image
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

    h_crop = patch_h * num_patches_y
    w_crop = patch_w * num_patches_x
    image = image[:, :h_crop, :w_crop]

    unfolded = F.unfold(
        image.unsqueeze(0),
        kernel_size=(patch_h, patch_w),
        stride=(patch_h, patch_w),
    )
    # unfolded: (1, C*patch_h*patch_w, num_patches)
    patches = unfolded.view(
        channels, patch_h, patch_w, num_patches_y, num_patches_x
    ).permute(
        3, 4, 0, 1, 2
    )  # [h, w, C, ph, pw]

    # Create normalized coordinates [-1, 1]
    y_coords = torch.linspace(-1, 1, num_patches_y)
    x_coords = torch.linspace(-1, 1, num_patches_x)

    xy = torch.stack(
        torch.meshgrid(x_coords, y_coords, indexing="xy"), -1
    )  # [grid_w, grid_h, 2]
    xy = xy.permute(1, 0, 2)  # [grid_h, grid_w, 2], match patch order
    return xy, patches  # patches[i] = one puzzle piece, xy[i] = where it belongs


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

    def __init__(self, images_path, txt_path, transform=None):
        self.images_path = images_path
        self.transform = transform

        self.image_names = []
        with open(txt_path, "r", encoding="utf-8") as f:
            self.image_names = f.read().splitlines()

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

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        image = self.dataset[idx]
        if self.image_size is not None:
            image = TF.resize(image, [self.image_size, self.image_size])
        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        if not isinstance(image, torch.Tensor):
            raise TypeError("Dataset must return a torch.Tensor or PIL.Image.")
        # image: [C, H, W]
        # grid-aware split
        coordinates_xy, patches = split_image_into_patches(
            image, self.num_patches_x, self.num_patches_y
        )

        # number_of_patches != grid_h * grid_w, multiple grid sizes, missing patches,padding / cropping,random resizing,rotation-only variants

        # Flatten patches and coordinates
        patches = einops.rearrange(patches, "h w c ph pw -> (h w) c ph pw")
        coordinates_xy = einops.rearrange(coordinates_xy, "h w c -> (h w) c")

        # Add rotation
        theta = torch.zeros(coordinates_xy.size(0), device=coordinates_xy.device)
        sin_cos = torch.stack([theta.sin(), theta.cos()], dim=-1)
        pose_gt = torch.cat([coordinates_xy, sin_cos], dim=-1)  # [N, 4]

        # Shuffle patches and pose_gt in the same way
        perm = torch.randperm(patches.size(0))
        patches = patches[perm]
        pose_gt = pose_gt[perm]

        # Optionally drop some patches
        if self.drop_ratio > 0:
            Number_patches = patches.size(0)
            remaining_patches = int((1 - self.drop_ratio) * Number_patches)
            patches = patches[:remaining_patches]
            pose_gt = pose_gt[:remaining_patches]

        edge_index = fully_connected_edge_index(patches.size(0), patches.device)

        # Convert to graph data
        graph_data = Data(
            x=patches,
            edge_index=edge_index,
            pos=coordinates_xy,
            pose_gt=pose_gt,  # node features  # diffusion target
        )

        return graph_data


# Testing the CelebA_DataSet class
if __name__ == "__main__":
    dataset = CelebA_DataSet(
        images_path="data/CelebA-HQ/images/CelebAMask-HQ/CelebA-HQ-img/",
        txt_path="data/CelebA-HQ/CelebA-HQ_test.txt",
    )
    print(f"Dataset size: {len(dataset)}")
    sample_img = dataset[0]
    print(f"Sample image size: {sample_img.size}")

    graph_dataset = CelebA_Graph_Dataset(dataset, num_patches_x=6, num_patches_y=6)
    graph = graph_dataset.get(0)
    print(f"Graph nodes: {graph.num_nodes}")
    print(f"Graph edges: {graph.edge_index.size(1)}")
    print(f"x shape: {graph.x.shape}")
    print(f"pose_gt shape: {graph.pose_gt.shape}")
