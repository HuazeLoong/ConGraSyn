import os
import sys
import csv
import torch
import numpy as np
from .const import *
from itertools import islice
from torch_geometric import data as DATA
from torch_geometric.data import Data
from PrepareData.Graph_data import GlobalVar

# GlobalVar.dist_bar = [3.0, 6.0]
from torch_geometric.data import InMemoryDataset
import h5py
import pandas as pd
from collections import defaultdict


class MyTestDataset(InMemoryDataset):
    def __init__(
        self,
        root="/tmp",
        dataset="_drug1",
        xd=None,
        xd_fp=None,
        xd_smi=None,
        xd_mol=None,
        xd_graph=None,
        xt=None,
        y=None,
        transform=None,
        pre_transform=None,
    ):
        self.cell2id = self.load_cell2id(CELL_ID_DIR)
        self.testcell = np.load(CELL_FEA_DIR)

        super(MyTestDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            self.data, self.slices = torch.load(
                self.processed_paths[0], weights_only=False
            )
            print("Use existing data files")
        else:
            self.process(xd, xd_fp, xd_smi, xd_mol, xd_graph, xt, y)
            self.data, self.slices = torch.load(
                self.processed_paths[0], weights_only=False
            )
            print("Create a new data file")

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + ".pt"]

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def load_cell2id(self, cell2id_file):
        cell2id = {}
        with open(cell2id_file, "r") as file:
            csv_reader = csv.reader(file, delimiter="\t")
            next(csv_reader)  # Skip the header
            for row in csv_reader:
                cell2id[row[0]] = int(row[1])
        return cell2id

    def get_cell_feature(self, cellId):
        return self.testcell[self.cell2id[cellId]] if cellId in self.cell2id else False

    def get_data(self, slice):
        d = [self.data[i] for i in slice]
        return MyTestDataset(d)

    @staticmethod
    def convert_graph(graph_dict):
        data = Data()
        num_nodes = len(graph_dict["atomic_num"])

        def to_tensor(key, dtype=torch.long, default=0, dim=1):
            val = graph_dict.get(key, None)
            if val is None:
                val = [default] * num_nodes
            elif len(val) < num_nodes:
                val = val + [default] * (num_nodes - len(val))
            elif len(val) > num_nodes:
                val = val[:num_nodes]
            return torch.tensor(val, dtype=dtype).unsqueeze(1 if dim == 1 else 0)

        # ===== 节点与边结构 =====
        data.atomic_num = to_tensor("atomic_num")
        data.chiral_tag = to_tensor("chiral_tag")
        data.is_aromatic = to_tensor("is_aromatic")
        data.pos = torch.tensor(graph_dict["atom_pos"], dtype=torch.float)
        data.edge_index = (
            torch.tensor(graph_dict["edges"], dtype=torch.long).t().contiguous()
        )
        num_edges = data.edge_index.size(1)

        # ===== 节点度数 =====
        degree = torch.zeros(num_nodes, dtype=torch.long)
        for i in range(num_edges):
            degree[data.edge_index[0, i]] += 1
        data.degree = degree.unsqueeze(1)

        # ===== 额外原子属性 =====
        for key in [
            "explicit_valence",
            "formal_charge",
            "num_explicit_Hs",
            "hybridization",
            "total_numHs",
            "atom_is_in_ring",
        ]:
            setattr(data, key, to_tensor(key))

        # ===== 原子级 attention 掩码 =====
        mask = torch.ones(num_nodes, dtype=torch.long)
        # pad 到 M
        M = getattr(GlobalVar, "max_nodes", num_nodes)
        pad_mask = torch.zeros(M, dtype=torch.long)
        pad_mask[:num_nodes] = mask
        data.atom_attention_mask = pad_mask

        # ===== 拼接为 data.x：统一维度 [num_nodes, feature_dim] =====
        feature_list = [
            data.atomic_num,
            data.chiral_tag,
            data.is_aromatic,
            data.degree,
            data.explicit_valence,
            data.formal_charge,
            data.num_explicit_Hs,
            data.hybridization,
            data.total_numHs,
            data.atom_is_in_ring,
        ]
        data.x = torch.cat(feature_list, dim=1)  # -> [num_nodes, 10]

        # ===== pair_distances: [N, N] =====
        pd = graph_dict.get("pair_distances", np.zeros((num_nodes, num_nodes)))
        if isinstance(pd, list):
            pd = np.array(pd)
        # pad 到最大节点数
        M = getattr(GlobalVar, "max_nodes", num_nodes)
        pd_padded = np.zeros((M, M), dtype=pd.dtype)
        pd_padded[:num_nodes, :num_nodes] = pd
        data.pair_distances = torch.tensor(pd_padded, dtype=torch.float)

        # ===== bond_distances: [num_edges] =====
        bd = graph_dict.get("bond_distances", np.zeros((num_edges,)))
        if isinstance(bd, list):
            bd = np.array(bd)
        # pad 到最大边数
        E = getattr(GlobalVar, "max_edges", num_edges)
        bd_padded = np.zeros((E,), dtype=bd.dtype)
        bd_padded[: min(len(bd), E)] = bd[: min(len(bd), E)]
        data.bond_distances = torch.tensor(bd_padded, dtype=torch.float)

        # ===== 功能团结构信息（若存在） =====
        data.FG_id = to_tensor("FG_id", default=-1)
        data.FG_mask = torch.tensor(
            graph_dict.get("FG_mask", [0] * num_nodes), dtype=torch.bool
        )
        data.FG_edge_index = (
            torch.tensor(graph_dict.get("FG_edge_index", []), dtype=torch.long)
            .t()
            .contiguous()
            if graph_dict.get("FG_edge_index")
            else torch.empty((2, 0), dtype=torch.long)
        )

        # ===== 原子距离桶参考值（若存在） =====
        if hasattr(GlobalVar, "dist_bar") and GlobalVar.dist_bar is not None:
            data.atom_dist_bar = torch.tensor(GlobalVar.dist_bar, dtype=torch.float)
        else:
            data.atom_dist_bar = torch.zeros(20, dtype=torch.float)  # 备用值

        return data

    def load_smi2idx(self):
        """
        构建 SMILES → 行号 的映射，用于直接从 h5 中提取表达谱。
        """
        screening_path = os.path.join(PERTUR_DIR, "drugcom_screening.csv")
        df = pd.read_csv(screening_path)
        return {smi: i for i, smi in enumerate(df["canonical_smiles"])}

    def process(self, xd, xd_fp, xd_smi, xd_mol, xd_graph, xt, y):
        assert len(xd) == len(xd_smi) and len(xt) == len(xt) and len(xt) == len(y)
        data_list = []
        slices = [0]

        smi2idx = self.load_smi2idx()

        for i in range(len(xd)):
            drug = xd[i]  # drug smiles
            drug_fp = xd_fp[i]  # fingerprints
            drug_smi = xd_smi[i]  # smiles atom token
            drug_mol = xd_mol[i]  # smiles mol
            graph_data = self.convert_graph(xd_graph[i])  # graph
            target = xt[i]  # cell line
            labels = y[i]  # label

            cell = self.get_cell_feature(target)
            if cell is False:
                print("Cell feature2 not found for target:", target)
                sys.exit()

            # Processing cell features
            if isinstance(cell, list) and isinstance(cell[0], np.ndarray):
                new_cell = np.array(cell)
            else:
                new_cell = cell
            if new_cell.ndim == 1:
                new_cell = np.expand_dims(new_cell, axis=0)
            graph_data.cell = torch.FloatTensor(new_cell)

            # === 获取表达谱 perturbation ===
            try:
                idx = smi2idx[drug]
            except KeyError:
                print(f"❌ SMILES 不在 screening.csv 中: {drug}")
                sys.exit()

            h5_path = os.path.join(PERTUR_DIR, f"{target}_100.h5")
            if not os.path.exists(h5_path):
                print(f"❌ H5 文件不存在: {h5_path}")
                sys.exit()

            with h5py.File(h5_path, "r") as f:
                if "x2_pred_inferred" not in f:
                    print(f"❌ H5 文件中缺少 x2_pred_inferred 字段: {h5_path}")
                    sys.exit()
                expr_mat = f["x2_pred_inferred"][:]
                if idx >= expr_mat.shape[0]:
                    print(f"❌ SMILES 映射 idx 超出表达谱行数: idx={idx}, shape={expr_mat.shape}")
                    sys.exit()

                expr_row = expr_mat[idx]
                expr_row = np.nan_to_num(expr_row, nan=0.0, posinf=0.0, neginf=0.0)
                graph_data.perturbation = torch.tensor(expr_row, dtype=torch.float)

                # print("NaN in perturbation:", torch.isnan(graph_data.perturbation).any())

            # data.token = drug_smi
            # data.mol = drug_mol
            graph_data.fp = drug_fp
            graph_data.token = drug_smi.clone().detach()
            graph_data.mol = drug_mol.clone().detach()
            graph_data.y = torch.Tensor([labels])
            # if i == 389 or i == 390 or i == 391 or i == 392:
            #     print(f"\n==== DEBUG for sample {i} ====")
            #     print("graph_data.x:", graph_data.x.shape)
            #     for k, v in graph_data.items():
            #         if k != 'x' and torch.is_tensor(v):
            #             print(f"{k}: {v.shape}")

            data_list.append(graph_data)

        # print("样本数量:", len(data_list))
        # print("data.mol shape 示例:", data_list[0].mol.shape)  # 应为 torch.Size([130])

        print("\n========= 打印样本 =========")
        if len(data_list) > 0:
            sample = data_list[0]
            print(sample)
            print("sample.cell:", type(sample.cell), sample.cell.shape)
            print("sample.fp:", type(sample.fp), sample.fp.shape)
            print("sample.token:", type(sample.token), sample.token.shape)
            print("sample.mol:", type(sample.mol), sample.mol.shape)
            print(
                "sample.perturbation:",
                type(sample.perturbation),
                sample.perturbation.shape,
            )
            print("graph summary (main):", sample)
            print("data.x (atom features):", sample.x.shape)
            print("sample.y:", type(sample.y), sample.y.shape)
        else:
            print("⚠️ data_list is empty, check your input data.")
        print("===========================\n")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
