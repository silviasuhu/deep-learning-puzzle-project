import os
import random
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from src.dataset import CelebA_DataSet, CelebA_Graph_Dataset

out_dir = "/home/angels/Projects/deep-learning-puzzle-project/outputs"
os.makedirs(out_dir, exist_ok=True)

images_path = "data/CelebA-HQ/images/CelebAMask-HQ/CelebA-HQ-img/"
txt_path = "data/CelebA-HQ/CelebA-HQ_train.txt"

dataset = CelebA_DataSet(images_path=images_path, txt_path=txt_path)
idx = random.randrange(len(dataset))
img = dataset[idx]

# Save original image
orig_path = os.path.join(out_dir, f"sample_{idx}_orig.png")
img.save(orig_path)

# Build graph and make a shuffled puzzle image
graph_ds = CelebA_Graph_Dataset(
    dataset, num_patches_x=6, num_patches_y=6, image_size=192
)
# Use same index to relate to the same image
graph = graph_ds.get(idx)

patches = graph.x  # [N, C, ph, pw]
N, C, ph, pw = patches.shape

grid_h = graph_ds.num_patches_y
grid_w = graph_ds.num_patches_x

# Build a grid in the shuffled order (puzzle)
patches_grid = patches.view(grid_h, grid_w, C, ph, pw)
puzzle = patches_grid.permute(2, 0, 3, 1, 4).reshape(C, grid_h * ph, grid_w * pw)

puzzle_img = TF.to_pil_image(puzzle.clamp(0, 1))
puzzle_path = os.path.join(out_dir, f"sample_{idx}_puzzle.png")
puzzle_img.save(puzzle_path)

print("Saved:", orig_path)
print("Saved:", puzzle_path)
