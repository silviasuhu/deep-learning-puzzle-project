import os

import torch
import torch_geometric  # for graph data handling

from PIL import Image


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
