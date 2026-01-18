import os
import torch
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


# Testing the CelebA_DataSet class
if __name__ == "__main__":
    dataset = CelebA_DataSet(
        images_path="data/CelebA-HQ/images/CelebAMask-HQ/CelebA-HQ-img/",
        txt_path="data/CelebA-HQ/CelebA-HQ_test.txt",
    )
    print(f"Dataset size: {len(dataset)}")
    sample_img = dataset[0]
    print(f"Sample image size: {sample_img.size}")
