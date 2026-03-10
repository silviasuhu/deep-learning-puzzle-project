"""
Implementation of a Eff_GAT variant that computes edge features the cosine similarity of the border pixels of the patches.
These are computed each step through a 1D CNN + global pooling to remove position information bias
and make it a fixed size vector.
Returns:
    _type_: _description_
"""
import timm
import torch
from torch import nn
from torch_geometric.nn import TransformerConv
from torch import Tensor
from torch_geometric.nn import GraphNorm
from torch.nn import functional as F

from .resnet_equivariant import ResNet18
from .gcn import GCN
from .Transformer_GNN import Transformer_GNN
from .transformerGNN_edgefeats import EdgeTransformer_GNN
from torchvision.transforms.functional import rotate

import itertools

# Get trimming slice for borders of the patches
def _valid_trim(length: int, trim: int) -> slice:
    if trim < 0:
        raise ValueError("trim must be >= 0")
    if 2 * trim >= length:
        raise ValueError(f"trim={trim} too large for border length={length}")
    return slice(trim, length - trim) if trim > 0 else slice(None)

# Get the edge pixels of each patch (after rescaling + dividing image into patches) to use as edge features.
def extract_patch_sides(patches: Tensor, trim: int = 0) -> Tensor:
    """
    Returns sides in fixed order [top, right, bottom, left].

    patches: [N, C, H, W]
    out: [N, 4, C, L]
    """
    if patches.ndim != 4:
        raise ValueError(f"Expected [N, C, H, W], got {tuple(patches.shape)}")

    _, _, h, w = patches.shape
    if h != w:
        raise ValueError("This implementation assumes square patches (H == W).")

    s = _valid_trim(h, trim)

    top = patches[:, :, 0, s]      # [N, C, L]
    right = patches[:, :, s, -1]   # [N, C, L]
    bottom = patches[:, :, -1, s]  # [N, C, L]
    left = patches[:, :, s, 0]     # [N, C, L]

    return top, right, bottom, left 


# Class to generate edge embeddings (1D conv + global pooling) from the edge pixels of the patches. 
# This is used to remove position information bias and get a fixed size edge feature vector.
class BorderEmbedding(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, kernel_size=3):
        super().__init__()
        # 1D convolution: in_channels=1 because we flatten all edges into a single channel
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.activation = nn.GELU()
        # Global max pooling to get fixed-size vector
        self.pool = nn.AdaptiveMaxPool1d(1)  # output length = 1
        self.out_dim = hidden_dim

    def forward(self, edge_vector):
        """
        edge_vector: [batch, length] -> flattened edges
        """
        x = edge_vector.unsqueeze(1)  # [batch, 1, length] -> 1 channel for conv1d
        x = self.pool(self.activation(self.conv1(x))) # [batch, hidden_dim, 1]
        x = x.squeeze(-1)             # [batch, hidden_dim]
        return x
    
############## MODEL ###############
class Border_GAT(nn.Module):
    """
    This model is Efficient_GAT variant that computes cosine similarity of the border pixels of the patches
    as edge features.


    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        steps,
        input_channels=2,
        output_channels=2,
        n_layers=4,
        visual_pretrained=True,
        freeze_backbone=False,
        model="resnet18equiv",
        architecture="edge_transformer",
        virt_nodes=4,
        all_equivariant=False,
    ) -> None:
        super().__init__()
        if model == "resnet18equiv":
            self.visual_backbone = ResNet18()
        else:
            self.visual_backbone = timm.create_model(
                model, pretrained=visual_pretrained, features_only=True
            )
        self.all_equivariant = all_equivariant
        self.model = model
        self.combined_features_dim = {
            "resnet18": 3136,
            "resnet50": 12352,
            "efficientnet_b0": 1088 + 32 + 32,
            "resnet18equiv": 1088 + 32 + 32,  # 3136,
            # 97792 + 32 + 32 resnet50
        }[model]

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.freeze_backbone = freeze_backbone

        if architecture == "edge_transformer":
            self.gnn_backbone = EdgeTransformer_GNN(
                self.combined_features_dim,
                n_layers=n_layers,
                hidden_dim=32 * 8,
                heads=8,
                output_size=self.combined_features_dim,
                edge_dim=32, # Output of the MLP processing the edge features
            )
        # elif architecture == "gcn":
        #     self.gnn_backbone = GCN(
        #         self.combined_features_dim,
        #         hidden_dim=32 * 8,
        #         output_size=self.combined_features_dim,
        #     )
        # elif architecture == "exophormer":
        #     self.gnn_backbone = Exophormer_GNN(
        #         self.combined_features_dim,
        #         n_layers=n_layers,
        #         hidden_dim=32 * 8,
        #         heads=8,
        #         output_size=self.combined_features_dim,
        #         virt_nodes=virt_nodes,
        #     )

        self.border_embedding = BorderEmbedding(in_channels=1, hidden_dim=64, kernel_size=3)
        self.border_mlp = nn.Sequential(
            nn.Linear(self.border_embedding.out_dim*2, 32), # 2 borders, 64 dims each, concat = 128
            nn.GELU(),
            nn.Linear(32,32)
        )

        self.time_emb = nn.Embedding(steps, 32)
        self.pos_mlp = nn.Sequential(
            nn.Linear(input_channels, 16), nn.GELU(), nn.Linear(16, 32)
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 32),
            nn.GELU(),
            nn.Linear(32, output_channels),
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.combined_features_dim),
        )

        self.linear1 = nn.Linear(8192, 544)  #  # dimension for resnet18

        self.linear2 = nn.Linear(4096, 544)  # dimension for resnet18

        mean = torch.tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        std = torch.tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, xy_pos, time, patch_rgb, edge_index, batch):
        patch_feats = self.visual_features(patch_rgb)
        border_feats = self.compute_edge_features(patch_rgb, edge_index)
        final_feats = self.forward_with_feats(
            xy_pos, time, patch_rgb, edge_index, border_feats=border_feats, patch_feats=patch_feats, batch=batch
        )
        return final_feats

    def forward_with_feats(
        self: nn.Module,
        xy_pos: Tensor,
        time: Tensor,
        patch_rgb: Tensor,
        edge_index: Tensor,
        border_feats: Tensor,
        patch_feats: Tensor,
        batch,
    ):

        time_feats = self.time_emb(time)  # embedding, int -> 32
        pos_feats = self.pos_mlp(xy_pos)  # MLP, (x, y) -> 32
        
        # COMBINE  and transform with MLP
        combined_feats = torch.cat([patch_feats, pos_feats, time_feats], -1)
        combined_feats = self.mlp(combined_feats)

        # GNN
        feats, attentions = self.gnn_backbone(
            x=combined_feats, 
            edge_index=edge_index, 
            edge_feats= border_feats,
            batch=batch
        )

        # Residual + final transform
        final_feats = self.final_mlp(feats + combined_feats)

        return final_feats, attentions

    # Compute cosine similairty of the embeddings of the edge pixels to use as edge features. 
    def compute_edge_features(self, patches, edge_index):
        # patches: [N, C, H, W]
        # edge_index: [2, E]
        n, c, h, w = patches.shape
        device = patches.device

        # Extract borders in batch 
        top = patches[:, :, 0, :]          # [N, C, L]
        right = patches[:, :, :, -1]       # [N, C, L]
        bottom = patches[:, :, -1, :]      # [N, C, L]
        left = patches[:, :, :, 0]         # [N, C, L]
        borders = torch.stack([top, right, bottom, left], dim=1)   # [N, 4, C, L]
        borders = borders.flatten(2)                              # [N, 4, C*L]

        # 2) Embed all borders in one batch
        border_flat = borders.reshape(n * 4, -1)                  # [N*4, C*L]
        emb = self.border_embedding(border_flat)                  # [N*4, H]
        emb = emb.view(n, 4, -1)                                  # [N, 4, H]

        # 3) Cosine similarities for edges in batch
        src, dst = edge_index
        emb_i = emb[src]   # [E, 4, H]
        emb_j = emb[dst]   # [E, 4, H]

        emb_i_n = F.normalize(emb_i, dim=-1)
        emb_j_n = F.normalize(emb_j, dim=-1)

        # Compare similarity using einsum (4x4 similrity matrix)
        sims = torch.einsum("eah,ebh->eab", emb_i_n, emb_j_n)      # [E, 4, 4]
        sims_flat = sims.view(sims.size(0), -1)                    # [E, 16]
        
        # Select best pair and extract original embeddings of those 2 borders
        best = sims_flat.argmax(dim=-1)
        bi = best // 4
        bj = best % 4
        idx = torch.arange(sims.size(0), device=device)
        best_bi = emb_i[idx, bi]                                   # [E, H]
        best_bj = emb_j[idx, bj]                                   # [E, H]
        
        # Stack and pass through a MLP
        edge_feat = self.border_mlp(torch.cat([best_bi, best_bj], dim=-1))
        
        return edge_feat
    
    def visual_features(self, patch_rgb):
        patch_rgb = (patch_rgb - self.mean) / self.std

        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.visual_backbone.forward(patch_rgb)
        else:
            if self.all_equivariant:
                feats = [
                    self.visual_backbone.forward(patch_rgb[:, i, :, :, :])
                    for i in range(4)
                ]
                feats = [
                    (feats[1][i] + feats[2][i] + feats[3][i] + feats[0][i]) / 4
                    for i in range(len(feats[1]))
                ]

            else:
                feats = self.visual_backbone.forward(patch_rgb)
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
            "resnet18equiv": [
                feats[2].reshape(patch_rgb.shape[0], -1),
                feats[3].reshape(patch_rgb.shape[0], -1),
            ],
        }[self.model]

        patch_feats = torch.cat(feats, -1)
        return patch_feats