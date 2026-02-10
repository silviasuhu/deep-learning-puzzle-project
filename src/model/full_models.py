import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from .groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4, P4MConvZ2

import timm
from torch import Tensor
from torch_geometric.nn import GraphNorm

from .gcn import GCN
from .exophormer_gnn import Exophormer_GNN
from .Transformer_GNN import Transformer_GNN
from torchvision.transforms.functional import rotate

from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn.conv.transformer_conv import TransformerConv
from torch_scatter import scatter


###################################################
################ BLOCKS ###########################
###################################################

################ ResNet blocks for the equivariant ResNet backbone ################
# This is a variant of the ResNet basic block for group-equivariant convolutions 
# in the P4 group (0º, 90º, 180º, 270º)
# Main idea is that rotating the input by n*90º produces a predictable rotation
# in the feature space, which will be useful to to learn for puzzles with rotations
class BasicBlock(nn.Module):
    expansion = 1 # Output channels are expanded by x1 (no expansion)
    def __init__(self, 
                 in_planes, # Input channels
                 planes, # Output channels
                 stride=1): #Stride
        super(BasicBlock, self).__init__() 
        # P4ConvP4 is a specific type of Group Equivariant Convolution (G-Conv) used in CNNs
        #  to handle data with rotational symmetry. 
        #  It operates on the p4 group, which consists of 90-degree rotations and translation
        self.conv1 = P4ConvP4(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes) # Becase we have a "rotation" dimension, we need 3d batch norm
        self.conv2 = P4ConvP4(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential() # The default shortcut (for residual) is the identity (no modification)

        # If spatial or channel dimension changes, we do a projection shortcut (residual)
        # 1x1 convolution + BatchNorm to match dimensions
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                P4ConvP4(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # Conv --> BN --> Relu
        out = self.bn2(self.conv2(out)) # Conv --> BN
        out += self.shortcut(x) # Add residual (either the identity (nothing) or the computed one)
        out = F.relu(out) # Relu
        return out

# Basic Bottleneck module (for ResNet)
class Bottleneck(nn.Module):
    expansion = 4 # Output channels are expanded by x4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # Same as above, define the P4ConvP4 convolution and BatchNorm layers
        self.conv1 = P4ConvP4(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = P4ConvP4(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = P4ConvP4(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion*planes)

        # Same as above, the (optional) residual 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                P4ConvP4(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # Conv --> BN --> Relu
        out = F.relu(self.bn2(self.conv2(out))) # Conv --> BN --> Relu
        out = self.bn3(self.conv3(out)) # Conv --> BN
        out += self.shortcut(x) # Add residual
        out = F.relu(out) # Relu
        return out

################ Exomorpher functions for the Exomorpher backbone ################
# Just a function that returns the activation function and corresponding expansion factor for the FFN hidden layer
def get_activation(activation):
    if activation == "relu":
        return 2, nn.ReLU()
    elif activation == "gelu":
        return 2, nn.GELU()
    elif activation == "silu":
        return 2, nn.SiLU()
    elif activation == "glu":
        return 1, nn.GLU()
    else:
        raise ValueError(f"activation function {activation} is not valid!")



###################################################
################ MODELS ###########################
###################################################

# Actual Resnet Equivariant variant model
class ResNet(nn.Module):
    def __init__(self, 
                 block, # Which block to ux
                 num_blocks, #  Number of blocks to build in each of the 4 "layers" --> [n1,n2,n3,n4]
                 num_classes=10 # Actually just unused in this implementation. In normal resnet, 
                                # used for classification, it's the number of output classes
                 ):
        super(ResNet, self).__init__()
        self.in_planes = 32 # Initialize channel width

        ###### IMPORTANT, EQUIVARIANCE EXPLANATION #########
        # First layer feeds the image (Z2 --> input channels = 3 channels -RGB-) to the P4 feature space  (32=4*8)
        # This is the key of the "equivariant" part of this Resnet
        # Z2 is the group a normal image is --> ie: equivariant to translation only in the 2D grid
        # P4 is a group that is equivariant to rotations in 4 discrete positions (0º, 90º, 180º, 270º) + translation 
        # So, OVERALL:
        # 1.- We rotate the patches during dataset creation in a discrete way
        # 2.- We pass them through this equivariant ResNet
        # 3.- This produces features that change PREDICTABLY with the rotation
        # 4.- This way, the feature reflect "how the patch change when X rotation is applied"
        # 5.- When feeding this features to the GNN (and the ground truth rotation as target), the GNN learns HOW
        # features reflect rotation for the patches
        # 6.- Finally, the GNN uses the learned information to predict rotations that need to be applied to each patch
        # when no ground truth (known rotation) is provided
        self.conv1 = P4ConvZ2(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes) # 3D Normalization (HxWxRotation)

        # 4 layers, each with the given number of residual or bottleneck blocks
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1) # First layer, with n1 blocks
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2) # Second layer, with n2 blocks
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2) # Third layer, with n3 blocks
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2) # Fourth layer, with n4 blocks
        #self.linear1 = nn.Linear(256*4*block.expansion, num_classes)
        self.linear1 = nn.Linear(128*4*block.expansion*32, 544)  # FC layer --> Projection head for out3 features
        self.linear2 = nn.Linear(256*4*block.expansion*8, 544) # FC layer --> Projection head for out4 features

    # Function to create each of the above layers
    # Gets the type and nubmer of blocks, stride, and planes (input channels) and stacks them updating the num channels (planes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) # Only the first block might have a diff stride (s=2), the rest are s=1
        layers = [] # Initialize list of blocks
        for stride in strides: # iterative styacking
            layers.append(block(self.in_planes, planes, stride)) # Append block to "layers" with the charact of the block
            self.in_planes = planes * block.expansion # Update planes/channel width
        return nn.Sequential(*layers) # Return a nn.Sequential with the stack of blocks

    # Main forward pass
    def forward(self, x):
        self.conv1(x) # This is unassigned, seems an artifact

        out = F.relu(self.bn1(self.conv1(x))) # Conv1 (converts Z2 image to P4) --> BN --> Relu
        out1 = self.layer1(out) # First layer of blocks
        out2 = self.layer2(out1) # Second layer of blocks
        out3 = self.layer3(out2)    # Third layer of blocks
        out4 = self.layer4(out3) # Forth layer of blocks
        
        #outs = out4.size()
        #out = out4.view(outs[0], outs[1]*outs[2], outs[3], outs[4])
        #out = F.avg_pool2d(out, 4)
        #print(out.size())
        #out = out.view(out.size(0), -1)

        # Now we get the flatten vector feature of out3 and out4
        # The backbone models like Efficient_GAT (see below) actually concatenate out3+out4 and use them both as combined feature
        # I guess the idea is to get 2 different levels of encoded features per each patch
        out3 = self.linear1(out3.view(out3.size(0), -1)) # Flatten out3 and project to 544
        out4 = self.linear2(out4.view(out4.size(0), -1)) # Flatten out4 and project to 544
        

        # output layer
        #out = self.linear(out)
        return [out1, out2, out3, out4] # Return intermediate features (out1, out2) and projected head vectors (oput3, out4)


#########################################################################################################
# These are honestly very silly ways to call predetermined Resnet Equivariant version that they might have tried.
# We can make our own

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])
#########################################################################################################

# Transformer backbone model
# Plain, simple, and Basic transformer GNN model, enough for small puzzles. 
# For larger puzzles (most of them, all in the implementation in the paper), Exomorpher is used
class Transformer_GNN(nn.Module):
    def __init__(self, 
                 input_size, # Dimension of input node features (after concat visual + pos + time feat)
                 hidden_dim, # Hidden size inside the transformer layers (number of channels in the transformer conv layers)
                 heads, # Number of attention heads
                 output_size, # Dimension of output node features (same as input, but for predictions)
                 n_layers=4) -> None: # Number of transformer layers to stack
        super().__init__()

        # Builds a stack of Convolutional transformer layers from PyG
        self.module_list = nn.ModuleList(
            [TransformerConv(input_size, out_channels=hidden_dim // heads, heads=heads)] # First layer: input_size --> hidden_dim
            + 
            [
                TransformerConv(
                    hidden_dim, out_channels=hidden_dim // heads, heads=heads # middle layers (n-2): hidden_dim --> hidden_dim
                )
                for _ in range(n_layers - 2)
            ]
            + 
            [
                TransformerConv( # Final layer: hidden_dim --> output_size, with concat of the heads (ie: heads*output_size//heads = output_size)
                    hidden_dim,
                    heads=heads,
                    concat=True,
                    out_channels=output_size // heads,
                ),
            ]
        )

        self.n_layers = n_layers # Assign num layers 

    # forward pass
    def forward(self, 
                x, # Node features (after concat visual + pos + time feat) --> [N, input_size]
                edge_index, # Edge index (ie: graph structure in PyG format) --> [2,E] -symmetric edges-
                move_to_cpu=False, # Whether to move output to CPU (if we want to visualize)
                batch=None, # Unused batch arg
                *args): # Other args, also not used
        attentions = [] # Initialize attention storage list
        for i in range(self.n_layers - 1): # Iterate through all but last layer
            x, atts = self.module_list[i]( # Pass through specific transformer layer and get output feats and attentions
                x=x, edge_index=edge_index, return_attention_weights=True
            )
            x = nn.functional.gelu(x) # Activation function after each layer (except last one)
            attentions.append(atts) # Append attentions for visualization

        x, atts = self.module_list[-1](         # Last layer, computed separately to avoid activation function
            x=x, edge_index=edge_index, return_attention_weights=True
        )
        attentions.append(atts) # Append last attentions

        # Move to CPU is flag is up
        if move_to_cpu:
            attentions = [(a[0].cpu().numpy(), a[1].cpu().numpy()) for a in attentions]
            x = x.cpu()
        return x, attentions # Return final output features and attentions

###################################################################################################################################################
# Exomorpher backbone model, main model used in the DiffAssemble paper
# It's basically a variant Transfomer GNN with virtual nodes to capture global graph info (useful for large puzzles)
# BRIEF EXPLANATION OF VIRTUAL NODES:
# For large puzzles, a fully connected graph becomes too expensive. In practice, a sparse graph with --degree=60% is used
# (which means for each graph of the batch, each node contains only 60% of the possible edges to other nodes, randomly selected at each epoch)
# This means each node is connected only to 60% of the other nodes, which makes harder to capture global information
# as it requires many "hops" to reach distant "directly unconnected" nodes
# To solve this, Exomoprher creates a series of vritual nods, each connecting to ALL real nodes
# For each graph of the batch, k virtual nodes are added, connected to all real nodes of that graph
# Now every real node can access global infor from virtual nodes even in a sparse graph (real node --> virtual node --> real node)
# Effectively allowing global info access in just 2 hops without the need of a densely connected graph
# Summary: dense graph scales O(N2), virtual nodes scale linearly O(N*k) where k is the number of virtual nodes
# They effetively act as a "summary of the graph" that can be accessed by all nodes

class Exophormer_GNN(nn.Module):
    def __init__(
        self, 
        input_size, # Dimension of input node features (after concat visual + pos + time feat)
        hidden_dim, # Hidden size inside the transformer layers (number of channels in the transformer conv layers)
        heads, # Number of attention heads
        output_size, # Dimension of output node features (same as input, but for predictions)
        n_layers=4, # Number of transformer layers to stack
        virt_nodes=4 # Number of virtual nodes to add per graph (virtual nodes are connected to ALL real nodes to capture global info)
    ) -> None:
        super().__init__()

        # Same as Transformer_GNN: stack of convolutional transformer layers
        self.module_list = nn.ModuleList(
            [TransformerConv(input_size, out_channels=hidden_dim // heads, heads=heads)] # First layer: input_size --> hidden_dim
            + 
            [
                TransformerConv(
                    hidden_dim, out_channels=hidden_dim // heads, heads=heads # middle layers (n-2): hidden_dim --> hidden_dim
                )
                for _ in range(n_layers - 2)
            ]
            + [
                TransformerConv( # Final layer: hidden_dim --> output_size, with concat of the heads (ie: heads*output_size//heads = output_size)
                    hidden_dim,
                    out_channels=output_size // heads,
                    heads=heads,
                    concat=True,
                ),
            ]
        )

        # Add virtual node embeddings
        self.virt_nodes = virt_nodes
        if self.virt_nodes > 0: # If vrt_nodes > 0, if not, it's ALMOST the same as Transformer_GNN (no identical due to GELU and attention handling)
            self.virt_node_embedding = nn.Embedding(virt_nodes, input_size) # learnable embeddings for virtual nodes
        self.n_layers = n_layers

    # Main forward pass
    def forward(self, 
                x, # Node features (after concat visual + pos + time feat) --> [N, input_size]
                edge_index, # Edge index (ie: graph structure in PyG format) --> [2,E] -symmetric edges-
                move_to_cpu=False, # Whether to move output to CPU (if we want to visualize)
                batch=None, # Here batch is used to identify which nodes belong to which graph in the batch, 
                            # which is important to add virtual nodes per graph
                mean_value=False, # Whether to init virtual nodes with the mean of real nodes (not used in final implementation)
                *args):
        attentions = [] # Initialize attention storage list

        n_graphs = batch.max() + 1  # Number of graphs in the batch (assuming starts at 0)
        num_real_nodes = len(batch) # Number of real nodes in that batch (not counting virtual nodes)
        device = batch.device # Device for tensor operations (CPU or GPU)
        
        # Main addition to Exomorpher: virtual nodes block
        if self.virt_nodes > 0: # Again, just run if virtual_nodes is > 0
            # Adding k virtual nodes per graph
            virtual_nodes = torch.arange(self.virt_nodes).repeat(n_graphs).to(device)
            # options for virtual node features initialization
            if mean_value: # Initialize as mean value of real nodes (NOT USED IN FINAL IMPLEMENTATION), faster and stable (feterministic), but less flexible
                virt_nodes_h = x.mean(dim=0).unsqueeze_(0).repeat(self.virt_nodes*n_graphs, 1)
            else: # Learnable virtual node embeddings (used in final implementation). Needs training, so slower, but more flexible/expressive
                virt_nodes_h = self.virt_node_embedding(virtual_nodes)
            x = torch.cat((x, virt_nodes_h)) # Concatenate virtual node features with real node features
            batch = torch.cat( # Update batch: real nodes keep their batch indices, virtual nodes are added to the end with their corresponding batch index
                (batch, torch.arange(n_graphs).repeat(self.virt_nodes).to(device)) # type: ignore
            )
            
            virt_edges = [] # Initialize list to store edges between virtual nodes and real nodes
            for i in batch.unique(): # iterate through each graph in the batch
                num_nodes = len(batch[batch == i])  # Number of nodes in the current graph (including virtual nodes)
                virt_edge = ( # Create edges between virtual nodes and real nodes: for each virtual node, connect to all real nodes of the same graph
                    torch.arange(
                        num_real_nodes + i * self.virt_nodes,
                        num_real_nodes + (i + 1) * self.virt_nodes,
                    )
                    .repeat(num_nodes)
                    .to(device)
            )
                virt_edges.append(virt_edge) # Append indices to list

            virt_edges = torch.cat(virt_edges) # Concatenate all virtual edges in a single tensor
            src_edges = torch.cat([torch.arange(num_real_nodes).to(device), virt_edges]) # Source nodes for new edges
            dst_edges = torch.cat([virt_edges, torch.arange(num_real_nodes).to(device)]) # Target nodes for new edges
            edge_index = torch.hstack((edge_index, torch.stack((src_edges, dst_edges)))) # Update edge index adding the new virtual-real node edges

        # Pass through transformer layers, slightly different than Transformer_GNN
        for i in range(self.n_layers - 1): # Iterate through all but last layer
            x = self.module_list[i](x=x, edge_index=edge_index) # Pass through specific transformer layer and get output feats and attentions
            # No activation function (GELU) after each layer (according to Exomorpher paper, that does not work well with virtual nodes)
            # No attention storage (but we could add it, there is no technical reason not to do it)
            
        # Last layer: pass and grab attentions (for visualization if needed)
        x, atts = self.module_list[-1](
            x=x, edge_index=edge_index, return_attention_weights=True
        )
        # Remove virtual nodes so the output matches the number of real node, so:
        # 1.- Can be compared to ground truth
        # 2.- Can be fed to the final MLP for predictions in the main model (eg: Eff_GAT) without dimensionality issues 
        if self.virt_nodes > 0:
            x = x[:num_real_nodes]  # remove virtual nodes
            
        attentions.append(atts) # Sotres only the last attention

        # Move to CPU is flag is up
        if move_to_cpu:
            attentions = [(a[0].cpu().numpy(), a[1].cpu().numpy()) for a in attentions]
            x = x.cpu()
        return x, attentions # Return final output features and attentions (only 1)
    
###################################################################################################################################################
# Efficient GAT model, which requires (and combines) the Resnet equivariant backbone with a GNN (Transformer, GCN or Exophormer) backbone
# Implementation in paper:
# Non-rotation puzzles (baseline experiments): ResNet18 + Transformer GNN with fully connected graph
# Rotation puzzles (main experiments): ResNet18 + Exophormer GNN with sparse graph (60% degree) and 8 virtual nodes
class Eff_GAT(nn.Module):
    """
    This model has 45M parameters
    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        steps, # Number of diffusion steps (for time embedding)
        input_channels=2, # Dimension of the input node features (translation only --> 2 (x,y), translation+rotation --> 4 (x,y, sin, cos))
        output_channels=2, # Dimension of the output node features (same as input, but for predictions)
        n_layers=4, # Number of layers for the GNN backbone
        visual_pretrained=True, # Whether to use pretrained weights for the visual backbone
        freeze_backbone=False, # Freeze visual backbone weights during training
        model="efficientnet_b0", # Visual backbone to use (resnet18equi, resnet18...)
        architecture="transformer", # GNN architecture to use (transformer, gcn, exophormer)
        virt_nodes=4, # Virtual nodes for Exophormer (if used)
        all_equivariant=False # The all_euivariant flag, to compute all 4 rotations in resnetequi and average them
    ) -> None:
        super().__init__()
        
        # Load the visual model (ResNet18 is the equivariant defined above, but we can define out custom one)
        if model == "resnet18equiv":
            self.visual_backbone = ResNet18()
        # If not using the custom models defined above, load the model from timm (PyTorch predefined Image Models) with pretrained weights
        else:
            self.visual_backbone = timm.create_model(
                model, pretrained=visual_pretrained, features_only=True
            )
        self.all_equivariant=all_equivariant
        self.model = model
        self.combined_features_dim = {
            "resnet18": 3136,
            "resnet50": 12352,
            "efficientnet_b0": 1088 + 32 + 32,
            'resnet18equiv': 1088 + 32 + 32, #3136,
            #97792 + 32 + 32 resnet50
        }[model] # Feature dimension after concatenating visual feat + pos feat + time feat --> input dimension for the GNN backbone
                    # If we define our own custom variant, we will need to add according to the output dim of our model

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.freeze_backbone = freeze_backbone

        # Load the backbone GNN
        if architecture == "transformer": # used for small puzzles, but not enough for larger (5x5 and above), not used in final implementation
            self.gnn_backbone = Transformer_GNN(
                self.combined_features_dim,
                n_layers=n_layers,
                hidden_dim=32 * 8,
                heads=8,
                output_size=self.combined_features_dim,
            )
        elif architecture == "gcn": # Not used at all in the final implementation
            self.gnn_backbone = GCN(
                self.combined_features_dim,
                hidden_dim=32 * 8,
                output_size=self.combined_features_dim,
            )
        elif architecture == "exophormer": # Main model used, as most puzzles are 5x5 and above
            self.gnn_backbone = Exophormer_GNN(
                self.combined_features_dim,
                n_layers=n_layers,
                hidden_dim=32 * 8,
                heads=8,
                output_size=self.combined_features_dim,
                virt_nodes=virt_nodes
            )


        self.time_emb = nn.Embedding(steps, 32) # Time embedding layer, maps diffusion step (integer) to a 32-dim vector
        self.pos_mlp = nn.Sequential(
            nn.Linear(input_channels, 16), nn.GELU(), nn.Linear(16, 32)
        ) # Takes pose input (eg: [x,y, cos, sin]) and maps it to a 32-dim vector to conat with visual feats
        
        # self.GN = GraphNorm(self.combined_features_dim) # This would normalize graph features if activated

        # this final layer takes the output of the GNN (which has the same dimension as the combined features) and produces the final
        # output (denoised pose prediction). Output dimension is the same as the input (2 or 4 dim)
        self.final_mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 32),
            nn.GELU(),
            nn.Linear(32, output_channels),
        )
        
        # An MLP layer that mixes the COMBINED features (visual + pos + time) before feeding them to the GNN
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.combined_features_dim),
        )

        # In theory, intended to adjust the dimensions for resnet18equi features, but not actually used in the implementation
        self.linear1 = nn.Linear(8192, 544) #  # dimension for resnet18
        self.linear2 = nn.Linear(4096, 544)  # dimension for resnet18

        # Nornalization constants for the visual backbone (ImageNet std and mean), registered as buffers for GPU compatibility
        mean = torch.tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        std = torch.tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    # High-level forward step
    def forward(self, 
                xy_pos,  # Node positions (includes rotations) --> [N, input_channels]
                time,  # Diffusion timestep for each node --> [N]
                patch_rgb, # Patch image for each node --> [N, 4, C, H, W] (if all_equivariant) or [N, C, H, W] (if not)
                edge_index, # Graph edgess in PyG format --> [2, E]
                batch): # Batch indices for each node (for PyG batching)
        patch_feats = self.visual_features(patch_rgb) # Call the visual backbone to get visual features for the input patch
        final_feats = self.forward_with_feats(
            xy_pos, time, patch_rgb, edge_index, patch_feats=patch_feats, batch=batch
        ) # Call the low-level forward_with _feats step to run the backbone GNN and extract the final output features (denoised pose predictions)
        return final_feats

    # Lower-level forward step with precomputed visual features to run the GNN backbone
    def forward_with_feats(
        self: nn.Module,
        xy_pos: Tensor, # Node positions (includes rotations) --> [N, input_channels]
        time: Tensor, # Diffusion timestep for each node --> [N]
        patch_rgb: Tensor, # Patch image for each node --> [N, 4, C, H, W] (if all_equivariant) or [N, C, H, W] (if not)
        edge_index: Tensor, # Graph edgess in PyG format --> [2, E]
        patch_feats: Tensor, # precomuted visual feats of each node
        batch, # Batch indices for each node (for PyG batching)
    ):

        time_feats = self.time_emb(time)  # Embedding, int (diffusion timestep) -> 32-dim vector
        pos_feats = self.pos_mlp(xy_pos)  # MLP, positions (x, y, sin, cos) -> 32-dim vector
        # COMBINE  and transform with MLP
        combined_feats = torch.cat([patch_feats, pos_feats, time_feats], -1) # Concatenae visual + pos + time feat
        combined_feats = self.mlp(combined_feats) # Pass through an MLP to mix before feeding to the GNN

        # Run the GNN: pass the combined feats, along with the graph structure (edge_index), and batch info
        # Outputs features and attentions (this last one only for Transformer and Exophormer, not for GCN)
        feats, attentions = self.gnn_backbone(
            x=combined_feats, edge_index=edge_index, batch=batch
        )


        # Residual connection + final transform = final target output (denoised pose prediction)
        final_feats = self.final_mlp(
            feats + combined_feats)
        return final_feats, attentions

    # Function to extract visual features from the input patch using the visual backbone
    def visual_features(self, patch_rgb):
        patch_rgb = (patch_rgb - self.mean) / self.std # Normalize input patch with ImageNet mean and std

        # Freeze backbone weights if specified
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.visual_backbone.forward(patch_rgb)
        # Otherwise compute features normally
        else:
            if self.all_equivariant: # If all_equivariant flag is up, compute feats for all 4 rotations and avg them
                feats = [self.visual_backbone.forward(patch_rgb[:, i, :, :, :]) for i in range(4)] # Comprehension for all 4 rot fwd passes
                feats = [(feats[1][i] + feats[2][i] + feats[3][i] + feats[0][i])/4 for i in range(len(feats[1]))]
            else: # Otherwise, compute only a single fwd pass
                feats = self.visual_backbone.forward(patch_rgb)
                
        # Flatten features and select relevant layers (2 = out3, and 3 = out4) --> see ResNet model above
        # Not sure why the dictionary, they are all computed the same
        feats = {
            "efficientnet_b0": [
                feats[2].reshape(patch_rgb.shape[0], -1),
                feats[3].reshape(patch_rgb.shape[0], -1),
            ],
            "resnet50": [
                feats[2].reshape(patch_rgb.shape[0], -1),
                feats[3].reshape(patch_rgb.shape[0], -1),
            ],
            "resnet18": [
                feats[2].reshape(patch_rgb.shape[0], -1),
                feats[3].reshape(patch_rgb.shape[0], -1),
            ],
                
            "resnet18equiv":[
                feats[2].reshape(patch_rgb.shape[0], -1),
                feats[3].reshape(patch_rgb.shape[0], -1),
              ]
        }[self.model]
        
        patch_feats = torch.cat(feats, -1) # Concatenate out3 and out4 features for the final visual feat representation
        return patch_feats # And return :)
