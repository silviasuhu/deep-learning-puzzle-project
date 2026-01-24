import os
import torch
from torch.utils.data import Dataset
import torch_geometric  # for graph data handling
from PIL import Image


def split_image_into_patches(image, patch_size):
    """
    Split a CxHxW image (non-overlapping patches) and normalizes coordinates to [-1, 1]
    Centered at 0,0 -> centered,symmetric,numerically stable for diffusion models).
    """

    Channels, Height, Width = image.shape
    # Calculate grid size
    grid_h = Height // patch_size
    grid_w = Width // patch_size

    # We want to split along height and width so channles should be last. torch.Tensor.unfold(dim, size, step)
    image2 = image.permute(1, 2, 0)  # CxHxW -> HxWxC
    patches = image2.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)

    # Create normalized coordinates [-1, 1]
    y_coords = torch.linspace(-1, 1, grid_h)
    x_coords = torch.linspace(-1, 1, grid_w)

    xy = torch.stack(
        torch.meshgrid(x_coords, y_coords, indexing="xy"), -1
    )  # [grid_w, grid_h, 2]
    xy = xy.permute(1, 0, 2)  # [grid_h, grid_w, 2], match patch order
    return xy, patches


class CelebA_DataSet(torch.utils.data.Dataset):
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
    def __init__(self, dataset=None, num_patches_x=6, num_patches_y=6):
        super().__init__()

        assert dataset is not None, "Dataset must be provided"
        self.dataset = dataset

        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        img = self.dataset[idx]

        graph_data = torch_geometric.data.Data(
            x=None,  # Node features
            indexes=None,  # Node indices
            edge_index=None,  # Edge indices
        )
        # Placeholder for graph conversion logic
        # Here you would convert the image to a graph representation
        graph_data = torch_geometric.data.Data()  # Replace with actual graph data
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
