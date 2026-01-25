import os
import torch
import einops
from torch.utils.data import Dataset
from PIL import Image
import torch_geometric
from torch_geometric.data import Data


def split_image_into_patches(image, num_patches_x, num_patches_y):
    """
    Split a CxHxW image (non-overlapping patches) and normalizes coordinates to [-1, 1]
    Centered at 0,0 -> centered,symmetric,numerically stable for diffusion models).
    """

    channels, height, width = image.shape
    # Calculate patch size
    patch_h = height // num_patches_y
    patch_w = width // num_patches_x

    # We want to split along height and width so channles should be last. torch.Tensor.unfold(dim, size, step)
    image2 = image.permute(1, 2, 0)  # CxHxW -> HxWxC
    patches = image2.unfold(0, patch_h, patch_h).unfold(
        1, patch_w, patch_w
    )  # shape: [num_patches_y, num_patches_x, patch_h, patch_w, C]

    # Create normalized coordinates [-1, 1]
    y_coords = torch.linspace(-1, 1, num_patches_y)
    x_coords = torch.linspace(-1, 1, num_patches_x)

    xy = torch.stack(
        torch.meshgrid(x_coords, y_coords, indexing="xy"), -1
    )  # [grid_w, grid_h, 2]
    xy = xy.permute(1, 0, 2)  # [grid_h, grid_w, 2], match patch order
    return xy, patches  # patches[i] = one puzzle piece, xy[i] = where it belongs


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
    def __init__(self, dataset=None, num_patches_x=6, num_patches_y=6, drop_ratio=0.0):
        super().__init__()

        self.dataset = dataset
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        self.drop_ratio = drop_ratio

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        image = self.dataset[idx]  # [C,H,W]
        # grid-aware split
        coordinates_xy, patches = split_image_into_patches(
            image, self.num_patches_x, self.num_patches_y
        )

        # number_of_patches != grid_h * grid_w, multiple grid sizes, missing patches,padding / cropping,random resizing,rotation-only variants

        # Flatten patches and coordinates
        patches = einops.rearrange(patches, "h w ph pw c -> (h w) c ph pw")
        coordinates_xy = einops.rearrange(coordinates_xy, "h w c -> (h w) c")

        # Add rotation
        theta = torch.zeros(coordinates_xy.size(0))
        sin_cos = torch.stack([theta.sin(), theta.cos()], dim=-1)
        pose_gt = torch.cat([coordinates_xy, sin_cos], dim=-1)  # [N, 4]

        # Convert to graph data
        graph_data = Data(
            x=patches, pose_gt=pose_gt  # node features  # diffusion target
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
