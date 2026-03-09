"""
Implementation of a Transformer GNN method that accepts edge features.

Returns:
    _type_: _description_
"""
from torch import nn
from torch_geometric.nn import TransformerConv


class EdgeTransformer_GNN(nn.Module):
    def __init__(self, input_size, hidden_dim, heads, output_size, edge_dim, n_layers=4) -> None:
        super().__init__()

        self.module_list = nn.ModuleList(
            [TransformerConv(input_size, 
                             out_channels=hidden_dim // heads, 
                             heads=heads,
                             edge_dim=edge_dim)]
            + [
                TransformerConv(
                    hidden_dim, 
                    out_channels=hidden_dim // heads, 
                    heads=heads,
                    edge_dim=edge_dim
                )
                for _ in range(n_layers - 2)
            ]
            + [
                TransformerConv(
                    hidden_dim,
                    heads=heads,
                    concat=True,
                    out_channels=output_size // heads,
                    edge_dim=edge_dim
                ),
            ]
        )

        self.n_layers = n_layers

    def forward(self, x, edge_index, edge_feats=None, move_to_cpu=False, batch=None, *args):
        attentions = []
        for i in range(self.n_layers - 1):
            x, atts = self.module_list[i](
                x=x, 
                edge_index=edge_index,
                edge_attr=edge_feats, 
                return_attention_weights=True
            )
            x = nn.functional.gelu(x)
            attentions.append(atts)

        x, atts = self.module_list[-1](
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_feats,
            return_attention_weights=True
        )
        attentions.append(atts)

        if move_to_cpu:
            attentions = [(a[0].cpu().numpy(), a[1].cpu().numpy()) for a in attentions]
            x = x.cpu()
        return x, attentions
