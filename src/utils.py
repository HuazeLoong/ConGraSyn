import torch
import numpy as np

from math import sqrt
from torch import nn
from scipy import stats
from torch_geometric.data import Batch
from sklearn.metrics import auc, mean_absolute_error, roc_auc_score
from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    balanced_accuracy_score,
)
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    roc_auc_score,
    precision_score,
)


def remove_nan_label(pred, truth):
    nan = torch.isnan(truth)
    truth = truth[~nan]
    pred = pred[~nan]

    return pred, truth


def roc_auc(pred, truth):
    return roc_auc_score(truth, pred)


def rmse(pred, truth):
    # print(f"pred type: {type(pred)}, truth type: {type(truth)}")
    pred_tensor = torch.tensor(pred, dtype=torch.float32)
    truth_tensor = torch.tensor(truth, dtype=torch.float32)

    return torch.sqrt(torch.mean(torch.square(pred_tensor - truth_tensor)))
    # return nn.functional.mse_loss(pred,truth)**0.5


def mae(pred, truth):
    return mean_absolute_error(truth, pred)

func_dict = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "mse": nn.MSELoss(),
    "rmse": rmse,
    "mae": mae,
    "crossentropy": nn.CrossEntropyLoss(),
    "bce": nn.BCEWithLogitsLoss(),
    "auc": roc_auc,
}

def get_func(fn_name):
    fn_name = fn_name.lower()
    return func_dict[fn_name]

def save_AUCs(AUCs, filename):
    with open(filename, "a") as f:
        f.write(",".join(map(str, AUCs)) + "\n")

def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse

from torch_geometric.data import Batch
import torch

def custom_collate(data_list):
    # 1) 先让 PyG 把图拼好
    batch = Batch.from_data_list(data_list)

    # 2) 堆叠那些固定长的张量
    batch.fp  = torch.stack([d.fp  for d in data_list], dim=0)
    batch.mol = torch.stack([d.mol for d in data_list], dim=0)

    # 3) 把原子 attention mask pad 到 [B, M]，再扩成 [B,1,1,M]
    mask = torch.stack([d.atom_attention_mask for d in data_list], dim=0)  # [B, M]
    attn_mask = mask.unsqueeze(1).unsqueeze(2)                              # [B,1,1,M]

    # 4) **不要再用 atom_attention_mask**，直接放到一个全新的字段 attn_mask 上
    batch.attn_mask = attn_mask

    # 5) 其它 pad 张量一并堆叠
    batch.pair_distances = torch.stack([d.pair_distances for d in data_list], dim=0)
    batch.bond_distances = torch.stack([d.bond_distances for d in data_list], dim=0)
    batch.atom_dist_bar   = torch.stack([d.atom_dist_bar   for d in data_list], dim=0)

    # 6) 新增扰动表达谱向量
    batch.pert_expr = torch.stack([d.perturbation for d in data_list], dim=0)

    # print(f"[collate] B={len(data_list)}, attn_mask.shape={attn_mask.shape}")

    return batch

def compute_preformence(T, S, Y, best_auc, file, epoch, elapsed_time_str, train_loss, test_loss):
    """Calculate multiple classification metrics and save the result corresponding to the best AUC."""
    AUC = roc_auc_score(T, S)
    precision, recall, threshold = metrics.precision_recall_curve(T, S)
    PR_AUC = metrics.auc(recall, precision)
    BACC = balanced_accuracy_score(T, Y)
    tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
    TPR = tp / (tp + fn)
    PREC = precision_score(T, Y)
    ACC = accuracy_score(T, Y)
    KAPPA = cohen_kappa_score(T, Y)

    AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, elapsed_time_str, train_loss, test_loss]
    if best_auc < AUC:
        save_AUCs(AUCs, file)
        best_auc = AUC
    return best_auc, AUC