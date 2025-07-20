import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_func
from torch import Tensor
from ProcessorData.token_dicts import Token2Idx, NewDic
from torch.nn import BatchNorm1d
from Model.smiles_encoder import *
from Model.Net.FP_GNN_NET import FP_GNN_NET
from Model.graph import Scage
from PrepareData.Graph_data import CompoundKit
from PrepareData.Graph_data import GlobalVar

# print(">>> dist_bar at script start:", GlobalVar.dist_bar)


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # gate = sigmoid( W [f1; f2] + b )
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, f1: Tensor, f2: Tensor) -> Tensor:
        # f1,f2: (B, dim)
        z = torch.cat([f1, f2], dim=1)  # (B, 2*dim)
        alpha = torch.sigmoid(self.gate(z))  # (B, 1)
        return alpha * f1 + (1 - alpha) * f2  # (B, dim)


class MultiSyn(nn.Module):
    # def __init__(self, n_output=2, num_features_xt=768, dropout=0.2, output_dim=160, deg=None):
    def __init__(self, n_output=2, num_features_xt=1280, dropout=0.2, output_dim=160):
        super(MultiSyn, self).__init__()
        hid_dim = 300
        self.act = get_func("ReLU")

        self.fuser = GatedFusion(dim=128)

        self.initialize_weights()

        self.atom_model = TrfmSeq2seq(len(Token2Idx), 256, len(Token2Idx), 4)
        self.mol_model = TrfmSeq2seq(len(NewDic), 256, len(NewDic), 4)
        # self.fp_model = FP_GNN_NET(deg=deg)
        self.fp_model = FP_GNN_NET()

        self.pert_proj = nn.Sequential(nn.Linear(10174, 128), nn.ReLU())  # 128

        self.ln = nn.LayerNorm(10174)

        self.scage = Scage(
            mode="finetune",
            atom_names=CompoundKit.atom_vocab_dict.keys(),
            atom_embed_dim=512,  # 512
            num_kernel=128,  # 128
            layer_num=6,  # 6
            num_heads=8,  # 8
            hidden_size=128,  # 128
            graph_output_dim=128,  # 1024
        ).cuda()

        # cell features MLP
        self.reduction = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        # combined layers **
        # self.pred = torch.nn.Sequential(
        #     torch.nn.Linear(2560, 1024),
        #     BatchNorm1d(2048),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(1024, 256),
        #     BatchNorm1d(256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, n_output),
        # )

        # pred2
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(2560, 2048),
            BatchNorm1d(2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_output),
            torch.nn.ReLU(),  # pred3 去掉relu
        )

        # pred4
        # self.pred = torch.nn.Sequential(
        #     torch.nn.Linear(2560, 2048),
        #     BatchNorm1d(2048),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(2048, 1024),
        #     BatchNorm1d(1024),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(1024, 256),
        #      BatchNorm1d(256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, n_output),
        #     torch.nn.ReLU()
        # )

        # # pred5
        # self.pred = torch.nn.Sequential(
        #     torch.nn.Linear(2560, 2048),
        #     BatchNorm1d(2048),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(2048, 1024),
        #     BatchNorm1d(1024),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(1024, 512),
        #      BatchNorm1d(512),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(512, n_output),
        #     torch.nn.ReLU()
        # )

        # pred6
        # self.pred = torch.nn.Sequential(
        #     torch.nn.Linear(2560, 2048),
        #     BatchNorm1d(2048),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(2048, 256),
        #     BatchNorm1d(256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, n_output),
        #     torch.nn.ReLU()
        # )

        # pred7 use this **
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(1024, 256),
            # BatchNorm1d(1024),
            # torch.nn.ReLU(),
            # torch.nn.Linear(1024, 256),
            BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_output),
            torch.nn.ReLU(),
        )

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    def forward(self, data1, data2):
        # print(">>> GlobalVar.dist_bar =", GlobalVar.dist_bar)
        fp_a = self.fp_model(data1.fp)
        feature_a = self.scage(data1)
        feat_a = self.fuser(fp_a, feature_a)

        fp_b = self.fp_model(data2.fp)
        feature_b = self.scage(data2)
        feat_b = self.fuser(fp_b, feature_b)

        cell = F.normalize(data1.cell, 2, 1)  # 768
        cell = self.reduction(cell)  # 512
        # print("cell.shape:", cell.shape)

        # 处理细胞扰动表达

        pert_a = self.pert_proj(F.normalize(data1.pert_expr, 2, 1))
        pert_b = self.pert_proj(F.normalize(data2.pert_expr, 2, 1))

        cell_vector = torch.cat((cell, pert_a, pert_b), 1)  # 512 + 256 * 2

        # concat
        xc = torch.cat((feat_a, feat_b, cell_vector), 1)  # (B, 1024 * 2 + 512)
        # print("xc.shape:", xc.shape)
        xc = F.normalize(xc, 2, 1)
        out = self.pred(xc)
        return out
