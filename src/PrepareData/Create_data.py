import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# 设置路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from PrepareData.Graph_data import _process_smiles_without_label
from PrepareData.finerprints import *
from PrepareData.Graph_data import *
from PrepareData.smi_tokenizer import *
from ProcessorData.const import *
from ProcessorData.dataset import *
from ProcessorData.token_dicts import *

from rdkit import RDLogger  # ✅ 添加这行导入

# ✅ 屏蔽所有 RDKit 的警告输出（包括弃用提示）
RDLogger.DisableLog('rdApp.*')

def preprocess_unique_smiles(unique_smiles, max_len=130):
    print(f"唯一 SMILES 数量：{len(unique_smiles)}，开始统一预处理...")

    fp_dict = {}
    atom_dict = {}
    mol_dict = {}
    graph_dict = {}

    for smi in tqdm(unique_smiles, desc="预处理 SMILES"):
        try:
            fp_dict[smi] = get_fingerprint(smi)
            atom_dict[smi] = torch.tensor(smi_tokenizer(smi, max_len=max_len, padding=True)).to(torch.int)
            mol_array = get_array([split(smi.strip())]).squeeze(0)
            # print("mol_array", mol_array.shape)
            mol_dict[smi] = mol_array
            graph_dict[smi] = _process_smiles_without_label(smi)  # 单个处理
        except Exception as e:
            print(f"预处理失败 SMILES: {smi}, 错误: {e}")

    return fp_dict, atom_dict, mol_dict, graph_dict

def creat_data(datafile):
    data_file = os.path.join(DATA_DIR, datafile)
    df = pd.read_csv(data_file + ".csv")
    
    # ---------------------- 新增：扫描最大节点数和最大边数 ----------------------
    # print("预处理：扫描最大节点数和最大边数...")
    # all_smiles = set(df["drug1_smiles"]).union(df["drug2_smiles"])
    # # 先预处理所有图，拿到它们的原子数和边数
    # temp_graphs = [_process_smiles_without_label(smi) for smi in all_smiles]
    # max_nodes = max(len(g['atomic_num']) for g in temp_graphs)
    # max_edges = max(len(g['edges']) for g in temp_graphs)
    # # 存到全局变量里，让 convert_graph 能访问
    # GlobalVar.max_nodes = max_nodes
    # GlobalVar.max_edges = max_edges
    # # 显式 str() 转换:
    # print("扫描最大节点数: " + str(GlobalVar.max_nodes) + "\n最大边数: " + str(GlobalVar.max_edges))
    GlobalVar.max_nodes = 69
    GlobalVar.max_edges = 72
    # --------------------------------------------------------------------------

    print("正在划分数据...")
    drug1, drug2, cell, label = (
        np.asarray(df["drug1_smiles"]),
        np.asarray(df["drug2_smiles"]),
        np.asarray(df["cell"]),
        np.asarray(df["label"]),
    )

    print("统一提取所有 SMILES")
    all_smiles = set(drug1.tolist() + drug2.tolist())
    fp_dict, atom_dict, mol_dict, graph_dict = preprocess_unique_smiles(all_smiles)

    print("\n正在映射特征到样本...")
    print("正在处理分子指纹...")
    fp1 = torch.stack([fp_dict[smi] for smi in drug1])
    fp2 = torch.stack([fp_dict[smi] for smi in drug2])
    print("分子指纹处理完毕！")

    print("正在处理原子指纹...")
    drug1_tensor = torch.stack([atom_dict[smi] for smi in drug1])
    drug2_tensor = torch.stack([atom_dict[smi] for smi in drug2])
    print("原子特征处理完毕！")

    print("正在处理分子特征...")
    xid1 = torch.stack([mol_dict[smi] for smi in drug1])
    xid2 = torch.stack([mol_dict[smi] for smi in drug2])
    print("分子特征处理完毕！")

    print("正在处理分子图...")
    graph1 = [graph_dict[smi] for smi in drug1]
    graph2 = [graph_dict[smi] for smi in drug2]
    print("分子图处理完毕！")

    print("数据维度：")
    print("药物数量 =", drug1.shape, "；SMILES token 向量 =", drug1_tensor.shape,
        "；指纹 =", fp1.shape, "；分词向量 =", xid1.shape,
        "分子图1 数量 =", len(graph1), "；分子图2 数量 =", len(graph2),
        "；细胞 =", cell.shape, "；标签 =", label.shape)

    MyTestDataset(
        root=DATAS_DIR,
        dataset=datafile + "_drug1",
        xd=drug1,
        xd_fp=fp1,
        xd_smi=drug1_tensor,
        xd_mol=xid1,
        xd_graph=graph1,
        xt=cell,
        y=label,
    )
    MyTestDataset(
        root=DATAS_DIR,
        dataset=datafile + "_drug2",
        xd=drug2,
        xd_fp=fp2,
        xd_smi=drug2_tensor,
        xd_mol=xid2,
        xd_graph=graph2,
        xt=cell,
        y=label,
    )

if __name__ == "__main__":
    datafile_list = ["drugcom_12415"]
    for datafile in datafile_list:
        creat_data(datafile)


"""
========= 打印样本 =========
Data(cell=[1, 768], fp=[1489], token=[130], mol=[220], y=[1])
data.cell: <class 'torch.Tensor'> torch.Size([1, 768])
data.fp: <class 'torch.Tensor'> torch.Size([1489])
data.token: <class 'torch.Tensor'> torch.Size([130])
data.mol: <class 'torch.Tensor'> torch.Size([220])
data.y: <class 'torch.Tensor'> torch.Size([1])
"""