from pathlib import Path
from typing import Optional, Callable, List
from PIL import Image
from PIL.Image import Resampling
from torch.utils.data import Dataset

import os
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
from scipy.sparse.linalg import eigsh
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

############################################################################################################################################
#############################################
#### GENERAL FUNCTIONS FOR THE DATASET ######
#############################################

#Function to divide the image into patches
@torch.jit.script
def divide_images_into_patches(
    img, # img (tensor shaped (C,H,W))
    patch_per_dim: List[int],  # Number of batches per dimension ([rows, cols])
    patch_size: int # Patch size (in pizels)
    ) -> List[Tensor]:

    img2 = img.permute(1, 2, 0) # Change dimensions of image tensor ((C, H, W) -> (H, W, C))
    
    # Split into patches using unfold, creates a tensor of shape:
    # (num_patches_row, num_patches_col, C, patch_size, patch_size)
    # unfold creates a sliding window along each dimension (0,1 = cols, rows) with a step of patch_size, 
    # effectively creating non-overlapping patches
    patches = img2.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size) 
    
    # Create normalized coordinate grid for each patch center, normalized -1,1
    y = torch.linspace(-1, 1, patch_per_dim[0]) 
    x = torch.linspace(-1, 1, patch_per_dim[1])
    xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)

    #Returns xy (position grid of each patch) adn patches (the actual patch images as a tensor)
    return xy, patches

#########################################
### FUNCTIONS TO GENERATE THE GRAPH #####
#########################################

# High-level function to generate graph
def create_graph(
    patch_per_dim, # Number of batches per dimension ([rows, cols])
    degree, # Degree of the graph: -1 for fully connected, otherwise expander graph of given degree
    unique_graph # RNG seed or none for random graph generation
    ):
    
    patch_edge_index_dict = {} # Create an empty dictionary to store edge indices for each patch dimension
    
    #Loop over each patch dimension to create the graph edge indices
    for patch_dim in patch_per_dim: # (0,1 = cols, rows)
        # If fully connected
        if degree == -1: 
            num_patches = patch_dim[0] * patch_dim[1]
            adj_mat = torch.ones(num_patches, num_patches) # Adjacency matrix is a square matrix of size 
                                                            #  (num_patches, num_patches) filled with ones 
            edge_index, _ = adj_mat.nonzero().t().contiguous() #Returns indices of non-zero elements (which are the edges in the graph)
     
        # If not fully connected (we could delete this if we are not gonna use non-fully connected graphs)
        else:
            num_patches = patch_dim[0] * patch_dim[1]
            edge_index = (
                generate_random_expander(
                    num_nodes=num_patches, degree=degree, rng=unique_graph
                )
                .t()
                .contiguous()
            )
            
        patch_edge_index_dict[patch_dim] = edge_index #Store the edge index in the dictionary with the patch dimension as the key
    return patch_edge_index_dict #Return dictionary containing indices for both patch dimensions (0,1 = cols, rows)

######## THE REST OF THE FUNCTIONS IN THIS SECTION ARE FOR GENERATING NON-FULLY CONNECTED GRAPHS ###########
# This generates a random d-regular expander graph with n nodes, used when we don't want full connectivity
# Requires more functions: get_eigenvalues() and generate_random_regular_graph()
# Expander: special type of sparse graph characterized by being exceptionally well-connected despite having very few edges
def generate_random_expander(num_nodes, degree, rng=None, max_num_iters=5, exp_index=0):
    """Generates a random d-regular expander graph with n nodes.
    Returns the list of edges. This list is symmetric; i.e., if
    (x, y) is an edge so is (y,x).
    Args:
      num_nodes: Number of nodes in the desired graph.
      degree: Desired degree.
      rng: random number generator
      max_num_iters: maximum number of iterations
    Returns:
      senders: tail of each edge.
      receivers: head of each edge.
    """

    # This allows to use a precentage (given as a string like 20%), 
    # and converts it into an integer degree based in the number of possible neighbors
    if isinstance(degree, str):
        degree = round((int(degree[:-1]) * (num_nodes - 1)) / 100)
    num_nodes = num_nodes # I don't think this actually does anything at all

    # If no seed is passed, generate one using Numpy
    if rng is None:
        rng = np.random.default_rng()
    
    # This is stuff about eigen values I dont quite get
    eig_val = -1
    eig_val_lower_bound = (
        max(0, degree - 2 * math.sqrt(degree - 1) - 0.1) if degree > 0 else 0
    )  # allow the use of zero degree

    max_eig_val_so_far = -1 # Initialize "best eigenvalue" as an "invalid" negative one
    max_senders = [] # Best source nodes of the graph
    max_receivers = [] # Best target nodes of the graph
    cur_iter = 1 # Iteration counter

    # (bave): This is a hack.  This should hopefully fix the bug
    # Avoid invalid degree values
    if num_nodes <= degree:
        degree = num_nodes - 1 # By basically overwritting tjhe input degree to num_nodes - 1 (parece un poco patillero esto)

    # Trick to avoid failure: if number of nodes is too small, justa add the whole graph even if you specified otherwise
    if num_nodes <= 10:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j: #Avoid self-loops
                    max_senders.append(i)
                    max_receivers.append(j)
    
    # If the size is adequate, just generate an actual normal random regular graph
    # Basically tries multiple times to generate  random expander graph and gets the best one (based on the seond lowest eigenvalue)
    else:
        while eig_val < eig_val_lower_bound and cur_iter <= max_num_iters: # Try multiple random regular graphs until hit thresholds
            senders, receivers = generate_random_regular_graph(num_nodes, degree, rng) # Generate a random d-regular graph (see below)

            eig_val = get_eigenvalue(senders, receivers, num_nodes=num_nodes) # Get eigenvalues for that graph

            # This is just in case eigenvalue computation failed, print info for debugging
            # Doesn0t actually "do" anything
            if len(eig_val) == 0:
                print(
                    "num_nodes = %d, degree = %d, cur_iter = %d, mmax_iters = %d, senders = %d, receivers = %d"
                    % (
                        num_nodes,
                        degree,
                        cur_iter,
                        max_num_iters,
                        len(senders),
                        len(receivers),
                    )
                )
                eig_val = 0         

            # If not, gety the second lowest eigenvalue (the one that we use as a proxy for quality, see below) 
            else:
                eig_val = eig_val[0]

            # If the eigenvalue of this graph is better that the best current one
            if eig_val > max_eig_val_so_far:
                max_eig_val_so_far = eig_val # Replace the max eigenvalue
                max_senders = senders # Replace the senders for this graph's senders
                max_receivers = receivers # Replace the reveivers for this graph's reveivers

            cur_iter += 1 # Increment iteration value

    #Convert to tensor, concat and return edges
    max_senders = torch.tensor(max_senders, dtype=torch.long).view(-1, 1)
    max_receivers = torch.tensor(max_receivers, dtype=torch.long).view(-1, 1)
    expander_edges = torch.cat([max_senders, max_receivers], dim=1)
    return expander_edges

# Function to generate a random d‑regular undirected graph
# This will be fed to generate_random_expander(), which will call get_eigenvalue() to evaluate how well connected the graph is
def generate_random_regular_graph(num_nodes, degree, rng=None):
    """Generates a random d-regular connected graph with n nodes.
    Returns the list of edges. This list is symmetric; i.e., if
    (x, y) is an edge so is (y,x).
    Args:
      num_nodes: Number of nodes in the desired graph.
      degree: Desired degree.
      rng: random number generator
    Returns:
      senders: tail of each edge.
      receivers: head of each edge.
    """
    # Checks
    # A d‑regular graph is only possible if num_nodes * degree is even
    if (num_nodes * degree) % 2 != 0:
        raise TypeError("nodes * degree must be even")
    # If no seed is provided, generate a random one
    if rng is None:
        rng = np.random.default_rng()
    # If degree is 0, return an empty graph/edges (just don't do that, wtf)
    if degree == 0:
        return np.array([]), np.array([])
    
    nodes = rng.permutation(np.arange(num_nodes)) # Shuffle nodes
    num_reps = degree // 2 # Number of “rolls” for creating pairs. Each roll adds 2 edges per node (edges are later doubled for symmetry).
    num_nodes = len(nodes) # ?? Redundant reassign, seems a 3 am decision

    # Creates num_reps shifted versions of nodes, stacked together; then created the edge index
    ns = np.hstack([np.roll(nodes, i + 1) for i in range(num_reps)])
    edge_index = np.vstack((np.tile(nodes, num_reps), ns))

    # Even degree case
    if degree % 2 == 0:
        # Symmetric assignment of senders and receivers of the edges
        # (Makes the graph undirected by adding reverse edges (u→v and v→u))
        senders, receivers = np.concatenate(
            [edge_index[0], edge_index[1]]
        ), np.concatenate([edge_index[1], edge_index[0]])
        return senders, receivers # And return the symmetric edge list
    
    # Odd degree case (we need to generate one more)
    else:
        # Adds one extra matching between the first half and second half of the permuted nodes, 
        # giving each node one extra neighbor.
        edge_index = np.hstack(
            (edge_index, np.vstack((nodes[: num_nodes // 2], nodes[num_nodes // 2 :])))
        )
        # Symmetric assignment, as above
        senders, receivers = np.concatenate(
            [edge_index[0], edge_index[1]]
        ), np.concatenate([edge_index[1], edge_index[0]])
        return senders, receivers # And return the symmetric edge list


# Computes the smallest Laplacian eigenvalues of a graph
# Mathematically, the second smallest eigenvalue of the expander is a proxy for how "good" the expander is
# (here "good" meaning something like "how closely representative of the fully connected graph is")
# So we use it above to evaluate the created expander: to keep the best graph it finds
def get_eigenvalue(senders, # Tail/origin of each edge
                   receivers, # Head/target of each edge
                   num_nodes): # Number of nodes
    
    edge_index = torch.tensor(np.stack([senders, receivers])) # Creates an edge index using senders/receivers lists
    # Build the (unnormalized) graph Laplacian L = D - A 
    edge_index, edge_weight = get_laplacian(
        edge_index, None, normalization=None, num_nodes=num_nodes
    )
    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes) # Convert to spare matrix (scipy can decompose this)
    return eigsh(L, k=2, which="SM", return_eigenvectors=False) # Return the computed two smallest‑magnitude eigenvalues of L


########################################################################################################################################
########################################################
##### CLASSES AND FUNCTIONS FOR DATA AGUMENTATION ######
########################################################

#Crop and resize
class RandomCropAndResizedToOriginal(transforms.RandomResizedCrop):
    def forward(self, img):
        size = img.size
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, size, self.interpolation)
    
# Functions for augmentations
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

########################################################################################################################################
############################
##### DATASET CLASSES ######
############################

#General CelebA dataset class, only grabs images, DOES NOT convert to tensor or create graph
class CelebA_HQ(Dataset):
    """
    ONLY loads images.
    No patches. No graphs. No diffusion logic.
    """

    def __init__(self, dataset_path, train=True):
        super().__init__()
        self.images_path = f"{dataset_path}/CelebAMask-HQ/CelebA-HQ-img"
        if train:
            txt_path = f"{dataset_path}/CelebA-HQ_train.txt"
        else:
            txt_path = f"{dataset_path}/CelebA-HQ_test.txt"

        self.image_names = []
        with open(txt_path, "r", encoding="utf-8") as f:
            self.image_names = f.read().splitlines()


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        return Image.open(os.path.join(self.images_path, self.image_names[idx]))


# General Puzzle dataset class
class Puzzle_Dataset(pyg_data.Dataset):
    def __init__(
        self,
        dataset=None, #Underlying dataset (eg: CelebA_HQ)
        patch_per_dim=[(7, 6)], #Patch sizes (eg: [(7,6)])
        patch_size=32,  # Size of each patch in pixels
        augment="", #Agumentation type ("" for none, "weak" or "hard")
        degree=-1, # Degree of the graph
        unique_graph=None, # RNG seed or none for random graph generation
        random=False, # If True, randomly shuffle patches
    ) -> None:
        super().__init__()

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

        # Creates an unique graph if the unique graph argument is a seed
        if self.unique_graph is not None:
            self.edge_index = create_graph(
                    self.patch_per_dim, self.degree, self.unique_graph
                )
    
    # Returns the length of the dataset
    def len(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)
        else:
            raise Exception("Dataset not provided")
        
    # Core function of the class: gets the image, divides into patches, creates the graph and returns the data object
    def get(self, idx): #Only input is the index of the image in the dataset
        if self.dataset is not None: # Ofc, only if the dataset is provided
            img = self.dataset[idx]

        #Picks a random patch dimension from the list provided (if only provided 1, it will always be the same)
        rdim = torch.randint(len(self.patch_per_dim), size=(1,)).item() # 
        patch_per_dim = self.patch_per_dim[rdim]

        height = patch_per_dim[0] * self.patch_size #Compute H of image based on patch_size and number of patches per that dimension
        width = patch_per_dim[1] * self.patch_size #Compute H of image based on patch_size and number of patches per that dimension
        img = img.resize((width, height)) # Rescale the image
        img = self.transforms(img) # Apply augmentations and convert to tensor

        xy, patches = divide_images_into_patches(img, patch_per_dim, self.patch_size) #get patches and coords using the function defined above

        #Flatten into node list
        xy = einops.rearrange(xy, "x y c -> (x y) c") 
        indexes = torch.arange(patch_per_dim[0] * patch_per_dim[1]).reshape(
            xy.shape[:-1]
        )
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")
        
        # Optional: Shuffle patches (random = True)
        if self.random:
            patches = patches[torch.randperm(len(patches))]
        
        # Build graph edges 
        if self.degree == -1: # Fully connected, seems to overwrite unique_graph argument
            adj_mat = torch.ones(
                patch_per_dim[0] * patch_per_dim[1], patch_per_dim[0] * patch_per_dim[1]
            )
            edge_index, _ = pyg.utils.dense_to_sparse(adj_mat)
            
        else:
            if not self.unique_graph: # Expander graph, only computed if unique_graph is False, otherwise it is precomputed in the constructor
                edge_index = generate_random_expander(
                    patch_per_dim[0] * patch_per_dim[1], self.degree
                ).T     

        
        # Pack into PyG Data object (PyTorch Geometric base Data class for graphs), with node features (xy), node indices (indexes), 
        # patch images (patches), edge indices (edge_index), image index (ind_name) and patch dimension (patches_dim)
        data = pyg_data.Data(
            x=xy,
            indexes=indexes,
            patches=patches,
            edge_index=self.edge_index[patch_per_dim]
            if self.unique_graph
            else edge_index,
            ind_name=torch.tensor([idx]).long(),
            patches_dim=torch.tensor([patch_per_dim]),
        )
        return data
    
    

# Missing Puzzle_Dataset_ROT and generate_random_expander
class Puzzle_Dataset_ROT(Puzzle_Dataset): # Constructed on top of Puzzle_Dataset
    def __init__(
        self,
        dataset=None,
        patch_per_dim=[(7, 6)],
        patch_size=32,
        augment=False,
        concat_rot=True, #  Whether to append rotation info to node features x.
        degree=-1, # Sparsity (-1 == 100% == fully connected)
        unique_graph=None,
        all_equivariant=False, # if true, keep all 4 rotated versions of each patch
        random_dropout=False, # Optional random edge subsampling
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
        # Get image with provided function
        if self.dataset is not None:
            img = self.dataset[idx]

        # All this are the same than the Puzzle_Dataset
        rdim = torch.randint(len(self.patch_per_dim), size=(1,)).item()
        patch_per_dim = self.patch_per_dim[rdim]

        height = patch_per_dim[0] * self.patch_size
        width = patch_per_dim[1] * self.patch_size

        img = img.resize((width, height), resample=Resampling.LANCZOS)#, resample=Resampling.BICUBIC)

        img = self.transforms(img)
        xy, patches = divide_images_into_patches(img, patch_per_dim, self.patch_size)

        xy = einops.rearrange(xy, "x y c -> (x y) c")
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")

        ## Here change a bit
        patches_num = patches.shape[0] # Number of patches

        # Converts patches to uint8 images
        patches_numpy = (
            (patches * 255).long().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
        )
        patches_im = [Image.fromarray(patches_numpy[x]) for x in range(patches_num)] # Wrap each patch as a PIL image, in a list
        random_rot = torch.randint(low=0, high=4, size=(patches_num,)) #Rotate each patch randomly 0=0º, 1=90º, 2=180º, 3=270º
        random_rot_one_hot = torch.nn.functional.one_hot(random_rot, 4) # Convert rotation labels into one-hot vectors


        # Create fully connected graph, if specified
        if self.degree == -1 or self.degree == "100%":
            adj_mat = torch.ones(
                patch_per_dim[0] * patch_per_dim[1], patch_per_dim[0] * patch_per_dim[1]
            )

            edge_index, _ = pyg.utils.dense_to_sparse(adj_mat)

        # Do some random droputs, if specified
        elif self.random_dropout:
            adj_mat = torch.ones(
                patch_per_dim[0] * patch_per_dim[1], patch_per_dim[0] * patch_per_dim[1]
            )

            edge_index, _ = pyg.utils.dense_to_sparse(adj_mat)
            degree = round(
                (int(self.degree[:-1]) * (int(patch_per_dim[0] * patch_per_dim[1]) - 1))
                / 100
            )
            n_connections = int(patch_per_dim[0] * patch_per_dim[1] * degree)
            edge_index = edge_index[:, torch.randperm(edge_index.shape[1])][
                :, :n_connections
            ]
        # Random expander graph (see above)
        else:
            if not self.unique_graph:
                edge_index = generate_random_expander(
                    patch_per_dim[0] * patch_per_dim[1], self.degree
                ).T

        # Create patch indices
        indexes = torch.arange(patch_per_dim[0] * patch_per_dim[1]).reshape(
            xy.shape[:-1]
        )

        # A map vector so rotations can be mapped 
        rots = torch.tensor(
            [
                [1, 0], # 0º
                [0, 1], # 90º
                [-1, 0], # 180º
                [0, -1], # 270º
            ]
        )

        # Map one-hot rotation to vectors
        rots_tensor = random_rot_one_hot @ rots

        # Using PIL, rotate each patch
        rotated_patch = [
            x.rotate(rot * 90) for (x, rot) in zip(patches_im, random_rot)
        ]  # in PIL

        # The damn equivariant part
        # If all_equivariant =True, then generate all 4 rotations per patch
        if self.all_equivariant:
            rotated_patch_1 = [
                [x.rotate(rot * 90) for rot in range(4)] for x in rotated_patch
            ]  # type: ignore <<<---- But this comment of theirs says "ignore", so I guess they don't use it at all

            rotated_patch_tensor = [
                [
                    torch.tensor(np.array(patch)).permute(2, 0, 1).float() / 255
                    for patch in test
                ]
                for test in rotated_patch_1
            ]

        else:
            # Convert back PIL images, rotated, to tensors
            rotated_patch_tensor = [
                torch.tensor(np.array(patch)).permute(2, 0, 1).float() / 255
                for patch in rotated_patch
            ]

        # Stack all into a tensor
        patches = (
            torch.stack([torch.stack(i) for i in rotated_patch_tensor])
            if self.all_equivariant
            else torch.stack(rotated_patch_tensor)
        )

        # If concat_rot=True, append rotation to node features (not sure if they use this)
        if self.concat_rot:
            xy = torch.cat([xy, rots_tensor], 1)

        # Pack into PyG Data object (PyTorch Geometric base Data class for graphs), with node features (xy), node indices (indexes), 
        # patch images (patches), edge indices (edge_index), image index (ind_name) and patch dimension (patches_dim)
        data = pyg_data.Data(
            x=xy,
            indexes=indexes,
            rot=rots_tensor,
            rot_index=random_rot,
            patches=patches,
            edge_index=self.edge_index[patch_per_dim]
            if self.unique_graph
            else edge_index,
            ind_name=torch.tensor([idx]).long(),
            patches_dim=torch.tensor([patch_per_dim]),
        )
        return data