import torch
import torch.nn as nn

from torch import nn
from Model.smiles_encoder import *
from Model.Net.PNA import PNA
from Model.mol_feature import MolFeature
from Model.Net.GIN_Net import GIN_Net
from torch_geometric.typing import OptTensor


def get_gnn_model(name, **kwargs):
    """
    @param name: model name in {GIN, MV_GNN_Graph, PNA}
    @return: gnn_model
    """
    if name == "GIN":
        model = GIN_Net(
            node_channels=MolFeature.get_atom_dim(),
            edge_channels=MolFeature.get_bond_dim(),
            hidden_channels=1024,
            out_channels=1024,
            num_layers=3,
            **kwargs
        )
    elif name == "PNA":
        model = PNA(
            in_channels=MolFeature.get_atom_dim(),
            hidden_channels=1024,
            out_channels=1024,
            num_layers=3,
            aggregators=["mean", "min", "max", "std"],
            scalers=["identity", "amplification", "attenuation"],
            # deg=kwargs.get("deg", None),
            edge_dim=MolFeature().get_bond_dim(),
            dropout=0.0,
        )
    else:
        raise Exception("没有这个 2d 网络")
    return model


class FPN(nn.Module):
    def __init__(
        self, in_channals, hidden_channals, mid_channals, out_channals, drop_p
    ):
        super(FPN, self).__init__()
        self.in_channals = in_channals
        self.lin1 = nn.Linear(in_channals, hidden_channals)
        self.lin2 = nn.Linear(hidden_channals, mid_channals)
        self.lin3 = nn.Linear(mid_channals, out_channals)
        # self.drop = nn.Dropout(p=drop_p)
        self.relu = nn.ReLU()

    def forward(self, fp):
        # (batchsize, fingerprints size)
        # fp = torch.concat([X.tsfm_fp.view(-1, 1024), X.traditional_fp.view(-1, 1489)], dim=1).to(torch.device('cuda'))
        # fp = fp.view(-1, self.in_channals)
        fp = fp.to(torch.device("cuda"))
        hidden_feature = self.lin1(fp)
        hidden_feature = self.relu(hidden_feature)
        out = torch.relu(self.lin2(hidden_feature))
        out = self.lin3(out)

        return out


class FP_GNN_NET(nn.Module):
    # def __init__(self, deg=None):
    def __init__(self):
        super(FP_GNN_NET, self).__init__()
        self.gcn_lin1 = nn.Linear(1024, 1024)
        self.fpn_model = FPN(
            in_channals=1489,
            hidden_channals=1024,
            mid_channals=1024,
            out_channals=1024,
            drop_p=0.0,
        )
        self.fcn_lin1 = nn.Linear(1024, 1024)
        self.hid_lin = nn.Linear(1024, 1024)
        self.out_lin = nn.Linear(1024, 128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.0)

    def forward(self, X):
        X = X.to(torch.device("cuda"))
        # gnn_out = self.gnn_model(X) # Graph-Level Encoder

        fpn_out = self.fpn_model(X)
        fpn_out = self.fcn_lin1(fpn_out)
        # gnn_out = torch.concat([gnn_out, fpn_out], dim=1)
        # out = gnn_out
        out = fpn_out
        out = self.hid_lin(self.relu(out))
        out = self.relu(out)
        out = self.drop(out)

        out = self.out_lin(out)

        return out
