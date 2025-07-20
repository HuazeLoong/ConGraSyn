import torch
import torch.nn as nn

from torch import nn
from torch.nn import ModuleList
from Model.mol_feature import MolFeature
from Model.smiles_encoder import *
from torch_geometric import nn as pnn
from Model.Layers.attention import Attention, GlobalAttention
from torch_geometric.nn import (
    BatchNorm,
    GINEConv,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)


class GIN_Net(nn.Module):
    def __init__(
        self,
        node_channels,
        edge_channels,
        hidden_channels,
        out_channels,
        num_layers,
        predict=False,
        num_classes=None,
    ):
        super().__init__()

        self.predict = predict
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.compress_pre_x = nn.Linear(256, 64)
        self.nn_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        if True:
            self.node_to_hid = nn.Linear(node_channels + 256, hidden_channels)
        else:
            self.node_to_hid = nn.Linear(node_channels, hidden_channels)
        self.attn_aggr = Attention(
            in_feature=out_channels, hidden=1024, out_feature=out_channels
        )

        for _ in range(num_layers):
            conv = GINEConv(nn=self.nn_layer, edge_dim=MolFeature().get_bond_dim())
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.lin1 = pnn.Linear(in_channels=hidden_channels, out_channels=1024)
        self.lin2 = pnn.Linear(in_channels=1024, out_channels=out_channels)
        if self.predict:
            self.linout = pnn.Linear(in_channels=out_channels, out_channels=num_classes)

        # if config.aggr_type == 'attention':
        #     self.readout = GlobalAttention(in_feature=out_channels,
        #                                    hidden_size=1024,
        #                                    out_feature=config.attn_outsize)
        #     self.att_lin1 = nn.Linear(in_features=config.attn_outsize * out_channels,
        #                               out_features=1024)
        #     self.att_lin2 = nn.Linear(in_features=1024,
        #                               out_features=out_channels)
        # elif config.aggr_type == 'sum':
        self.readout = global_add_pool
        # elif config.aggr_type == 'mean':
        #     self.readout = global_mean_pool
        # elif config.aggr_type == 'max':
        #     self.readout = global_max_pool
        # else:
        #     self.readout = global_add_pool

        self.device = torch.device("cuda")

    def forward(self, X):
        edge_index = X.edge_index
        if True:
            # print(f'x.shape: {X.x.shape} prex.shape: {X.pre_x.shape}')
            # compressed_pre_x = self.compress_pre_x(X.pre_x)
            node_feature = torch.concat([X.x, X.pre_x], dim=1)
        else:
            node_feature = X.x
        edge_feature = X.edge_attr
        edge_index = edge_index.to(self.device)
        node_feature = node_feature.to(self.device)
        batch = X.batch.to(self.device)
        edge_feature = edge_feature.to(self.device)

        # def forward(self, x: Tensor, edge_index: Adj,
        #             edge_attr: OptTensor = None) -> Tensor:
        hid = self.node_to_hid(node_feature)
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            hid = torch.relu(conv(x=hid, edge_index=edge_index, edge_attr=edge_feature))

        out = self.lin1(hid)
        out = torch.relu(out)
        out = self.lin2(out)
        # out = torch.relu(out)
        # out = self.linout(out)

        out = self.readout(out, batch)
        # if config.aggr_type == 'attention':
        #     out = self.att_lin2(torch.relu(self.att_lin1(out)))
        # out,  = self.attn_aggr(out)
        if self.predict:
            out = torch.relu(out)
            out = self.linout(out)

        return out
