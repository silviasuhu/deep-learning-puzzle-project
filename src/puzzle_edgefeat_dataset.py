"""
This file contains the implementation Puzzle Dataset with Edge Features class, a minimal derivative the 
Puzzle_Dataset class from DiffAssemble. This implementation is currently only designed to support fully 
connected graphs.

The original Dataset class from DiffAssemble is described originally as:
A PyTorch Geometric dataset for creating puzzle-like graph data from images. The dataset divides images 
into patches and constructs a graph where each patch is a node and edges are defined based on a specified 
degree of connectivity. The dataset also supports various augmentations, including random rotations and
missing pieces, to create more challenging puzzles for training graph neural networks.

Original: DiffAssemble repository:
(https://github.com/IIT-PAVIS/DiffAssemble/blob/release/puzzle_diff/dataset/puzzle_dataset.py)
"""

import math
import random
from typing import List, Tuple

import einops
import networkx as nx
import numpy as np
import torch
import torch_geometric as pyg
import torch_geometric.data as pyg_data
import torch_geometric.loader
import torchvision.transforms as transforms
from PIL import Image
from PIL.Image import Resampling
from scipy.sparse.linalg import eigsh
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


import torch.nn as nn


class RandomCropAndResizedToOriginal(transforms.RandomResizedCrop):
    def forward(self, img):
        size = img.size
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, size, self.interpolation)


def _get_augmentation(augmentation_type: str = "none"):
    switch = {
        "weak": [transforms.RandomHorizontalFlip(p=0.5)],
        "hard": [
            transforms.RandomHorizontalFlip(p=0.5),
            RandomCropAndResizedToOriginal(
                size=(1, 1), scale=(0.8, 1), interpolation=InterpolationMode.BICUBIC
            ),
        ],
    }
    return switch.get(augmentation_type, [])


@torch.jit.script
def divide_images_into_patches(
    img, patch_per_dim: List[int], patch_size: int
) -> List[Tensor]:
    # img2 = einops.rearrange(img, "c h w -> h w c")

    # divide images in non-overlapping patches based on patch size
    # output dim -> a
    img2 = img.permute(1, 2, 0)
    patches = img2.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    y = torch.linspace(-1, 1, patch_per_dim[0])
    x = torch.linspace(-1, 1, patch_per_dim[1])
    xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
    # print(patch_per_dim)

    return xy, patches


# generation of a unique graph for each number of nodes
def create_graph(patch_per_dim, degree, unique_graph):
    # Create an empty dictionary
    patch_edge_index_dict = {}
    for patch_dim in patch_per_dim:
        if degree == -1:
            num_patches = patch_dim[0] * patch_dim[1]
            adj_mat = torch.ones(num_patches, num_patches)
            edge_index, _ = adj_mat.nonzero().t().contiguous()
            
        ##  CURRENTLY ONLY FULLY CONNECTED GRAPHS ARE SUPPORTED
        else:
            raise ValueError("Unique graph generation is currently only supported for fully connected graphs (degree=-1)")
        patch_edge_index_dict[patch_dim] = edge_index
    return patch_edge_index_dict

class Puzzle_Dataset(pyg_data.Dataset):
    def __init__(
        self,
        dataset=None,
        patch_per_dim=[(7, 6)],
        patch_size=32,
        augment="",
        degree=-1,
        unique_graph=None,
        random=False,
    ) -> None:
        super().__init__()

        assert dataset is not None
        self.dataset = dataset
        self.patch_per_dim = patch_per_dim
        self.unique_graph = unique_graph
        self.augment = augment
        self.random = random

        self.transforms = transforms.Compose(
            [
                *_get_augmentation(augment),
                transforms.ToTensor(),
            ]
        )
        self.patch_size = patch_size
        self.degree = degree

        if self.unique_graph is not None:
            self.edge_index = create_graph(
                self.patch_per_dim, self.degree, self.unique_graph
            )

    def len(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)
            # return 100
        else:
            raise Exception("Dataset not provided")

    def get(self, idx):
        if self.dataset is not None:
            img = self.dataset[idx]

        rdim = torch.randint(len(self.patch_per_dim), size=(1,)).item()
        patch_per_dim = self.patch_per_dim[rdim]

        height = patch_per_dim[0] * self.patch_size
        width = patch_per_dim[1] * self.patch_size
        img = img.resize((width, height))  # , resample=Resampling.BICUBIC)
        img = self.transforms(img)

        xy, patches = divide_images_into_patches(img, patch_per_dim, self.patch_size)

        xy = einops.rearrange(xy, "x y c -> (x y) c")

        indexes = torch.arange(patch_per_dim[0] * patch_per_dim[1]).reshape(
            xy.shape[:-1]
        )
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")
        if self.random:
            patches = patches[torch.randperm(len(patches))]
        if self.degree == -1:
            # all connected to all
            adj_mat = torch.ones(
                patch_per_dim[0] * patch_per_dim[1], patch_per_dim[0] * patch_per_dim[1]
            )

            edge_index, _ = pyg.utils.dense_to_sparse(adj_mat)
            
            # Empty edge features (placeholder)
            edge_attr = torch.zeros(edge_index.shape[1], 1)  # 1 dummy feature
            
        else:
            raise ValueError("Unique graph generation is currently only supported for fully connected graphs (degree=-1)")
        
        data = pyg_data.Data(
            x=xy,
            indexes=indexes,
            patches=patches,
            edge_index=(
                self.edge_index[patch_per_dim] if self.unique_graph else edge_index
            ),
            edge_attr=edge_attr, #Add dummy edge attributes (computed during diffusion step)
            ind_name=torch.tensor([idx]).long(),
            patches_dim=torch.tensor([patch_per_dim]),
        )
        return data


class Puzzle_Dataset_ROT(Puzzle_Dataset):
    def __init__(
        self,
        dataset=None,
        patch_per_dim=[(7, 6)],
        patch_size=32,
        augment=False,
        concat_rot=True,
        degree=-1,
        unique_graph=None,
        all_equivariant=False,
        random_dropout=False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            patch_per_dim=patch_per_dim,
            patch_size=patch_size,
            augment=augment,
            degree=degree,
            unique_graph=unique_graph,
        )
        self.concat_rot = concat_rot
        self.degree = degree
        self.all_equivariant = all_equivariant
        self.unique_graph = unique_graph
        self.random_dropout = random_dropout
        if self.unique_graph is not None:
            self.edge_index = create_graph(
                self.patch_per_dim, self.degree, self.unique_graph
            )

    def get(self, idx):
        if self.dataset is not None:
            img = self.dataset[idx]

        rdim = torch.randint(len(self.patch_per_dim), size=(1,)).item()
        patch_per_dim = self.patch_per_dim[rdim]

        height = patch_per_dim[0] * self.patch_size
        width = patch_per_dim[1] * self.patch_size

        img = img.resize(
            (width, height), resample=Resampling.LANCZOS
        )  # , resample=Resampling.BICUBIC)

        img = self.transforms(img)
        xy, patches = divide_images_into_patches(img, patch_per_dim, self.patch_size)

        xy = einops.rearrange(xy, "x y c -> (x y) c")
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")

        patches_num = patches.shape[0]

        patches_numpy = (
            (patches * 255).long().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
        )
        patches_im = [Image.fromarray(patches_numpy[x]) for x in range(patches_num)]
        random_rot = torch.randint(low=0, high=4, size=(patches_num,))
        random_rot_one_hot = torch.nn.functional.one_hot(random_rot, 4)

        # if self.degree == '100%':

        if self.degree == -1 or self.degree == "100%":
            adj_mat = torch.ones(
                patch_per_dim[0] * patch_per_dim[1], patch_per_dim[0] * patch_per_dim[1]
            )

            edge_index, _ = pyg.utils.dense_to_sparse(adj_mat)
            
            # Empty edge features (placeholder)
            edge_attr = torch.zeros(edge_index.shape[1], 1)  # 1 dummy feature
            

        else:
            raise ValueError("Unique graph generation is currently only supported for fully connected graphs (degree=-1)")

        indexes = torch.arange(patch_per_dim[0] * patch_per_dim[1]).reshape(
            xy.shape[:-1]
        )

        rots = torch.tensor(
            [
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
            ]
        )

        rots_tensor = random_rot_one_hot @ rots

        # ruoto l'immagine casualmente

        rotated_patch = [
            x.rotate(rot * 90) for (x, rot) in zip(patches_im, random_rot)
        ]  # in PIL

        if self.all_equivariant:
            rotated_patch_1 = [
                [x.rotate(rot * 90) for rot in range(4)] for x in rotated_patch
            ]  # type: ignore

            rotated_patch_tensor = [
                [
                    torch.tensor(np.array(patch)).permute(2, 0, 1).float() / 255
                    for patch in test
                ]
                for test in rotated_patch_1
            ]
        else:
            rotated_patch_tensor = [
                torch.tensor(np.array(patch)).permute(2, 0, 1).float() / 255
                for patch in rotated_patch
            ]

        patches = (
            torch.stack([torch.stack(i) for i in rotated_patch_tensor])
            if self.all_equivariant
            else torch.stack(rotated_patch_tensor)
        )
        if self.concat_rot:
            xy = torch.cat([xy, rots_tensor], 1)

        data = pyg_data.Data(
            x=xy,
            indexes=indexes,
            rot=rots_tensor,
            rot_index=random_rot,
            patches=patches,
            edge_index=(
                self.edge_index[patch_per_dim] if self.unique_graph else edge_index
            ),
            edge_attr=edge_attr, #Add dummy edge attributes (computed during diffusion step)
            ind_name=torch.tensor([idx]).long(),
            patches_dim=torch.tensor([patch_per_dim]),
        )
        return data
