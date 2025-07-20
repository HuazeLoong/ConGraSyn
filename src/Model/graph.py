import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ProcessorData.const import *
from torch import Tensor
from torch.nn import Embedding
from typing import Dict, List, Tuple
from PrepareData.Graph_data import GlobalVar
from PrepareData.Graph_data import CompoundKit
from torch_geometric.utils import to_dense_batch
from Model.Layers.projection import AtomProjection, AtomFGProjection


class DEFAULTS:
    DROP_RATE = 0.1


# noinspection DuplicatedCode
class MultiScaleAttention(nn.Module):
    # def __init__(self, num_heads, hidden_dim, dropout=0.1, attn_dropout=0.1, temperature=1,
    #              use_super_node=True):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout=0.1,
        attn_dropout=0.1,
        temperature=1,
        use_super_node=True,
        knn_ratio: float = 0.2,  # 保留最近 α 比例 的邻居
    ):
        super(MultiScaleAttention, self).__init__()

        self.d_k = hidden_dim // num_heads
        self.num_heads = num_heads  # number of heads
        self.temperature = temperature
        self.use_super_node = use_super_node
        self.knn_ratio = knn_ratio

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.a_proj = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(len(GlobalVar.dist_bar))]
        )

        self.scale_linear = nn.Sequential(
            nn.Linear(len(GlobalVar.dist_bar) * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(1, num_heads, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(num_heads, num_heads, kernel_size=1),
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        # qkv_reset_parameters(self)

    def forward(
        self, x, dist, dist_bar: Tensor, mask=None, attn_bias=None, dist_embed=None
    ):
        # print(f"[DEBUG] dist_bar used in attention: {dist_bar.tolist()}")
        B, N_full, D = x.shape
        H, d_k = self.num_heads, self.d_k
        num_scales = dist_bar.size(0)

        # — QKV —
        query = self.q_proj(x).view(B, N_full, H, d_k).transpose(1, 2)  # [B,H,N,D']
        key = self.k_proj(x).view(B, N_full, H, d_k).transpose(1, 2)
        value = self.v_proj(x).view(B, N_full, H, d_k).transpose(1, 2)

        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if attn_bias is not None:
            #     scores = scores + attn_bias
            # print(">> attn_bias shape:", attn_bias.shape)
            b = attn_bias.unsqueeze(1)
            # print(">> b after unsqueeze(1)", b.shape)
            b = b.expand(-1, self.num_heads, num_scales, -1, -1)
            # print(">> b after expand:", b.shape)
            # print('>> scores before:', scores.shape)
            scores = scores.unsqueeze(2)  # [B,H,1,N,N]
            scores = scores.expand(-1, H, num_scales, -1, -1)  # [B,H,K,N,N]
            scores = scores + b
        if mask is not None:
            # mask: [B,1,N_full,N_full]
            # print(">> raw mask shape:", mask.shape)
            m = mask.unsqueeze(1)  # → [B,1,1,N,N]
            # print(">> after unsqueeze(1):", m.shape)
            m = m.expand(-1, self.num_heads, num_scales, -1, -1)  # → [B,H,S,N,N]
            # print(">> after expand:", m.shape)
            scores = scores.masked_fill(m == 0, -1e12)

        # — 重建 new_dist —
        dist = dist.unsqueeze(1)  # [B,1,N,N]
        n_node = dist.size(-1) + int(self.use_super_node)
        new_dist = dist.new_ones([B, 1, n_node, n_node])
        new_dist[:, :, int(self.use_super_node) :, int(self.use_super_node) :] = dist

        # print(new_dist.min(), new_dist.max(), dist_bar)
        # print((new_dist < dist_bar[0]).float().mean())

        # —— 打印阈值和距离范围 ——
        # print(">>> dist_bar:", dist_bar.tolist())
        # print(f">>> new_dist range: min={new_dist.min().item():.4f}, max={new_dist.max().item():.4f}")

        # 超节点不参与 mask，值设为 0
        new_dist[:, :, 0, :] = 0
        new_dist[:, :, :, 0] = 0

        # — 准备各阈值的统一 mask —
        # dist_bar: Tensor of shape [K], contains你要的 K 个阈值
        bars = dist_bar.view(1, 1, -1, 1, 1)  # [1,1,K,1,1]
        d = new_dist.view(B, 1, 1, n_node, n_node)  # [B,1,1,n,n]
        masks = d < bars  # [B,1,K,n,n]

        # 保证超节点 都为 True
        masks[..., :, 0, :] = True
        masks[..., :, :, 0] = True

        # —— 插入统计放行率 ——
        dm = masks.squeeze(1)  # [B, K, n_node, n_node]
        keep_list = []
        for k in range(num_scales):
            keep_pct = float(dm[:, k].float().mean()) * 100
            keep_list.append(keep_pct)
            # print(f">>> threshold {dist_bar[k].item():.2f}: keep {keep_pct:.1f}% of pairs")

        # 统计最小和最大保留率
        min_keep, max_keep = min(keep_list), max(keep_list)
        # print(f">>> overall coverage range: {min_keep:.1f}% – {max_keep:.1f}%")

        coverage_str = (
            f">>> overall coverage range: {min_keep:.1f}% – {max_keep:.1f}%\n"
        )
        # 追加写入到 coverage.log 文件
        log_path = os.path.join(RESULTS_DIR, "coverage.log")
        with open(log_path, "a") as f:
            f.write(coverage_str)

        # — 计算每个 branch 的 masked scores & attention —
        # scores: [B,H,n,n] → 加一个维度成 [B,H,1,n,n] 以便 broadcast
        # s = scores.unsqueeze(2)                        # [B,H,1,n,n]
        scores_masked = scores.masked_fill(~masks, -1e12)  # [B,H,K,n,n]
        attn = F.softmax(scores_masked, dim=-1)  # [B,H,K,n,n]
        attn = self.attn_dropout(attn)

        # — 消息传递，按阈值拼接 —
        # value: [B,H,n,D']; 首先 expand to [B,H,K,n,D']
        v = value.unsqueeze(2).expand(-1, -1, dist_bar.size(0), -1, -1)
        msg = torch.matmul(attn, v)  # [B,H,K,n,D']

        # 如果有 dist_embed，要同样 expand 并用 einsum:
        if dist_embed is not None:
            # dist_embed: [B,H,N,N,D]
            # 1) 在 scale 维前 unsqueeze，得到 6 维 [B,H,1,N,N,D]
            d_emb = dist_embed.unsqueeze(2)
            # 2) 显式 expand 到 [B,H,S,N,N,D]
            d_emb = d_emb.expand(-1, self.num_heads, num_scales, -1, -1, -1)
            # 3) einsum，计算每个 scale 上的 distance-message
            #    'bhkij,bhkijd->bhkid'  保留 i 维（节点），和 d 维（feature）
            dist_msg = torch.einsum("bhkij,bhkijd->bhkid", attn, d_emb)  # [B,H,S,N,D]

            # 4) v_msg 已经是 [B,H,S,N,D]，直接相加
            v_msg = torch.matmul(
                attn, value.unsqueeze(2).expand(-1, -1, num_scales, -1, -1)
            )
            msg = v_msg + dist_msg
        else:
            msg = torch.matmul(
                attn, value.unsqueeze(2).expand(-1, -1, num_scales, -1, -1)
            )

        # 最后沿 K 维拼到 head 维上
        msg = (
            msg.transpose(1, 2).contiguous().view(B, H * dist_bar.size(0), N_full, d_k)
        )
        msg = msg.transpose(2, 3).contiguous().view(B, N_full, -1)  # [B,N,D*K]
        x = self.scale_linear(msg) + x
        x = self.layer_norm(x)

        return x, attn  # 或返回 attn_list，如果你想调试每个阈值


class PositionWiseFeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim,
        ffn_hidden_dim,
        activation_fn="GELU",
        dropout=DEFAULTS.DROP_RATE,
    ):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.fc2 = nn.Linear(ffn_hidden_dim, hidden_dim)
        self.act_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn_act_func = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.dropout(self.fc2(self.act_dropout(self.ffn_act_func(self.fc1(x)))))
        x += residual
        x = self.ffn_layer_norm(x)
        return x


class MultiScaleTransformer(nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        ffn_hidden_dim,
        dropout=0.1,
        attn_dropout=0.1,
        temperature=1,
        activation_fn="GELU",
    ):
        super(MultiScaleTransformer, self).__init__()
        assert hidden_dim % num_heads == 0
        self.self_attention = MultiScaleAttention(
            num_heads, hidden_dim, dropout, attn_dropout, temperature
        )
        self.self_ffn_layer = PositionWiseFeedForward(
            hidden_dim, ffn_hidden_dim, activation_fn=activation_fn
        )

    def forward(self, x, dist, dist_bar, attn_mask, attn_bias=None, dist_embed=None):
        x, attn = self.self_attention(
            x,
            dist,
            dist_bar,
            mask=attn_mask,
            attn_bias=attn_bias,
            dist_embed=dist_embed,
        )
        x = self.self_ffn_layer(x)

        return x, attn


class EncoderAtomLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        ffn_hidden_dim,
        num_heads,
        dropout=DEFAULTS.DROP_RATE,
        attn_dropout=DEFAULTS.DROP_RATE,
        temperature=1,
        activation_fn="GELU",
    ):
        super(EncoderAtomLayer, self).__init__()
        self.transformer = MultiScaleTransformer(
            num_heads,
            hidden_dim,
            ffn_hidden_dim,
            dropout,
            attn_dropout,
            temperature,
            activation_fn,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer_norm.weight)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor,
        attn_bias: Tensor = None,
        dist=None,
        dist_bar=None,
        dist_embed=None,
    ):
        x, attn = self.transformer(x, dist, dist_bar, attn_mask, attn_bias, dist_embed)

        return x


def safe_index(alist, elem):
    return alist.index(elem) if elem in alist else len(alist) - 1


def rd_chem_enum_to_list(values):
    """values = {0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER}
    """
    return [values[i] for i in range(len(values))]


def clean_fg_dict_with_max_coverage_fn_group_priority(
    fg_dict: Dict[int, List[List[int]]]
) -> Dict[int, List[List[int]]]:
    """
    Clean functional group dictionary with max coverage functional group priority.
    That is, if an atom is in multiple functional groups, it will be assigned to the functional group with the largest
    number of atoms.

    However, the condition is that the atoms in the smaller functional group should be a subset of the atoms in the
    larger functional group.
    Args:
        fg_dict: functional group dictionary. {fn_id -> list of matches}. each match is a list of atom indices.
    Returns:
        cleaned functional group dictionary.
    """
    all_matches = set(tuple(match) for matches in fg_dict.values() for match in matches)
    for fn_id, matches in fg_dict.items():
        for idx, match in enumerate(matches):
            for other_match in all_matches:
                if set(match) < set(other_match):
                    matches[idx] = []
    fg_dict = {
        fn_id: [match for match in matches if match]
        for fn_id, matches in fg_dict.items()
    }
    return fg_dict


def get_fg_list(
    n_atom: int, fg_dict: Dict[int, List[List[int]]]
) -> Tuple[List[int], List[int]]:
    fg = [0 for _ in range(n_atom)]
    fg_index_arr = [0 for _ in range(n_atom)]
    for idx in range(n_atom):
        max_length = 0
        max_length_key = 0
        fg_index = 0
        for fn_id, matches in fg_dict.items():
            for i in range(len(matches)):
                if idx in matches[i] and len(matches[i]) > max_length:
                    max_length = len(matches[i])
                    max_length_key = fn_id
                    fg_index = len(CompoundKit.fg_mo_list) * (i + 1) + fn_id
        fg[idx] = max_length_key
        fg_index_arr[idx] = fg_index
    return fg, fg_index_arr


def match_fg(mol, edges) -> Tuple[List[int], List[int]]:
    n_atom = len(mol.GetAtoms())
    # fg_dict = {}
    fg_dict: Dict[int, List[List[int]]] = {}
    # fn_group_idx -> list of atom index that belong to the functional group
    idx = 0
    for sn in CompoundKit.fg_mo_list:
        matches = mol.GetSubstructMatches(sn)
        idx += 1
        # match_num = 0
        fg_dict.setdefault(idx, [])
        for match in matches:
            fg_dict[idx].append(list(match))
    fg_dict = clean_fg_dict_with_max_coverage_fn_group_priority(fg_dict)
    fg, fg_index_arr = get_fg_list(n_atom, fg_dict)
    edges_index = [
        0 if fg_index_arr[edges[i][0]] == fg_index_arr[edges[i][1]] else 1
        for i in range(len(edges))
    ]
    return fg, edges_index


@torch.jit.script
def gaussian(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    # noinspection PyTypeChecker
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (
        torch.sqrt(2 * torch.pi) * std
    )


# noinspection PyPep8Naming
class GaussianKernel(nn.Module):
    def __init__(
        self,
        K: int = 128,
        std_width: float = 1.0,
        start: float = 0.0,
        stop: float = 9.0,
    ):
        super().__init__()
        self.K = K
        mean = torch.linspace(start, stop, K)
        std = std_width * (mean[1] - mean[0])
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.mul = Embedding(1, 1, padding_idx=0)
        self.bias = Embedding(1, 1, padding_idx=0)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        mul = self.mul.weight
        bias = self.bias.weight
        x = (mul * x.unsqueeze(-1)) + bias
        expand_shape = [-1] * len(x.shape)
        expand_shape[-1] = self.K
        x = x.expand(expand_shape)
        mean = self.mean.float()
        return gaussian(x.float(), mean, self.std)


# noinspection PyUnresolvedReferences
class AtomEmbedding(nn.Module):
    def __init__(self, atom_names, embed_dim, num_kernel):
        super(AtomEmbedding, self).__init__()
        self.atom_names = atom_names

        self.embed_list = nn.ModuleList()
        for name in self.atom_names:
            embed = nn.Embedding(
                CompoundKit.get_atom_feature_size(name) + 5, embed_dim, padding_idx=0
            )
            self.embed_list.append(embed)

        self.graph_embedding = nn.Embedding(1, embed_dim)

        self.graph_finger_print = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.mass_embedding = nn.Sequential(
            GaussianKernel(K=num_kernel, std_width=1.0, start=0.0, stop=9.0),
            nn.Linear(num_kernel, embed_dim),
        )

        self.van_der_waals_radis_embedding = nn.Sequential(
            GaussianKernel(K=num_kernel, std_width=1.0, start=0.0, stop=9.0),
            nn.Linear(num_kernel, embed_dim),
        )

        self.partial_charge_embedding = nn.Sequential(
            GaussianKernel(K=num_kernel, std_width=1.0, start=0.0, stop=9.0),
            nn.Linear(num_kernel, embed_dim),
        )

        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, node_features: Dict[str, Tensor]):
        # print("当前 node_features keys:", node_features.keys())
        out_embed = 0
        for i, name in enumerate(self.atom_names):
            # print(f"正在处理原子特征: {name}")
            # if name not in node_features:
            #     print(f"⚠️ 警告: 缺失特征 {name}")
            out_embed += self.embed_list[i](node_features[name])
        if "mass" in node_features:
            mass_embed = self.mass_embedding(node_features["mass"])
            out_embed += mass_embed
        if "van_der_waals_radius" in node_features:
            van_der_waals_radis_embed = self.van_der_waals_radis_embedding(
                node_features["van_der_waals_radius"]
            )
            out_embed += van_der_waals_radis_embed
        if "partial_charge" in node_features:
            partial_charge_embed = self.partial_charge_embedding(
                node_features["partial_charge"]
            )
            out_embed += partial_charge_embed

        graph_token_embed = self.graph_embedding.weight.unsqueeze(0).repeat(
            out_embed.size()[0], 1, 1
        )

        out_embed = torch.cat([graph_token_embed, out_embed], dim=1)
        # normalize
        # out_embed = out_embed / (out_embed.norm(dim=-1, keepdim=True) + 1e-5)
        out_embed = self.final_layer_norm(out_embed)
        return out_embed


# noinspection PyPep8Naming,SpellCheckingInspection
class Scage(nn.Module):
    def __init__(
        self,
        mode,
        atom_names,
        atom_embed_dim,
        num_kernel,
        layer_num,
        num_heads,
        hidden_size,
        graph_output_dim=512,
    ):
        super(Scage, self).__init__()

        self.mode = mode
        self.atom_names = atom_names
        self.atom_embed_dim = atom_embed_dim
        self.num_kernel = num_kernel
        self.layer_num = layer_num
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.graph_output_dim = graph_output_dim

        self.register_buffer("dist_bar", torch.tensor(GlobalVar.dist_bar))

        D = atom_embed_dim  # e.g. 512

        # —— 新增 —— 坐标编码 MLP —— 把每个原子 (x,y,z) → D 维向量
        self.coord_mlp = nn.Sequential(nn.Linear(3, D), nn.ReLU(), nn.Linear(D, D))

        # —— 新增 —— 距离编码 MLP —— 把标量距离 → 一个 attention bias
        # 输出一个标量 bias（也可以输出多头维度再 reshape）
        self.edge_dist_mlp = nn.Sequential(
            nn.Linear(1, D // num_heads),  # 或者 1,1，根据你想插入 attn_bias 的方式决定
            nn.ReLU(),
            nn.Linear(D // num_heads, 1),
        )

        self.atom_feature = AtomEmbedding(
            self.atom_names, self.atom_embed_dim, self.num_kernel
        )

        self.EncoderAtomList = nn.ModuleList()
        for j in range(self.layer_num):
            self.EncoderAtomList.append(
                EncoderAtomLayer(self.atom_embed_dim, self.hidden_size, self.num_heads)
            )

        self.head_Graph = AtomProjection(
            self.atom_embed_dim, self.atom_embed_dim // 2, self.graph_output_dim
        )

    def forward(self, batched_data):
        # print(f"[SCAGE] dist_bar used = {self.dist_bar.tolist()}")
        # —— 原子特征 to_dense_batch ——
        # x_flat: [total_nodes, F=10], batch: [total_nodes]
        x_flat = batched_data.x
        x_dense, node_mask = to_dense_batch(x_flat, batched_data.batch)
        # x_dense: [B, N, 10], node_mask: [B, N]

        B, N, _ = x_dense.shape
        Np1 = N + 1  # +1 for the super-node

        # —— 构造 AtomEmbedding 所需的 feature dict ——
        atom_feats = x_dense
        node_features = {
            "atomic_num": atom_feats[:, :, 0].long(),
            "chiral_tag": atom_feats[:, :, 1].long(),
            "is_aromatic": atom_feats[:, :, 2].long(),
            "degree": atom_feats[:, :, 3].long(),
            "explicit_valence": atom_feats[:, :, 4].long(),
            "formal_charge": atom_feats[:, :, 5].long(),
            "num_explicit_Hs": atom_feats[:, :, 6].long(),
            "hybridization": atom_feats[:, :, 7].long(),
            "total_numHs": atom_feats[:, :, 8].long(),
            "atom_is_in_ring": atom_feats[:, :, 9].long(),
        }

        # —— 得到带 super-node 的初始 atom embedding ——
        # atom: [B, N+1, D]
        atom = self.atom_feature(node_features)

        # ===== 新增 1: 坐标编码 & 融合 =====
        # pos_flat: [total_nodes, 3]
        pos_flat = batched_data.pos
        pos_dense, _ = to_dense_batch(pos_flat, batched_data.batch)  # [B, N, 3]

        # MLP 编码 3D 坐标 → [B, N, D]
        coord_emb = self.coord_mlp(pos_dense)

        # 为 super-node 补零向量，再 concat → [B, N+1, D]
        coord_zero = torch.zeros(B, 1, coord_emb.size(-1), device=coord_emb.device)
        coord_emb = torch.cat([coord_zero, coord_emb], dim=1)

        # 把坐标嵌入加到 atom embedding 上
        atom = atom + coord_emb

        # ===== 新增 1 end =====

        # —— 构造 attention mask ——
        # super_mask = torch.ones((B, 1), device=node_mask.device, dtype=node_mask.dtype)
        # mask2d = torch.cat([super_mask, node_mask], dim=1)
        # attn_mask4d = mask2d.unsqueeze(1).unsqueeze(2)
        # 构造 node-level mask
        super_mask = torch.ones((B, 1), device=node_mask.device, dtype=node_mask.dtype)
        mask2d = torch.cat([super_mask, node_mask], dim=1)  # [B, N_full]

        # 构造 pairwise mask：只有两端节点都存在的对才有效
        # mask2d[:, :, None]: [B, N_full, 1], mask2d[:, None, :]: [B, 1, N_full]
        pair_mask = mask2d.unsqueeze(2) & mask2d.unsqueeze(1)  # [B, N_full, N_full]

        # 现在才是真正的 attention mask
        attn_mask4d = pair_mask.unsqueeze(1)  # [B,1,N_full,N_full]

        # —— 取 pair_distances & 裁剪 ——
        # pair_distances: [B, M, M], M >= N
        pair_dist = batched_data.pair_distances[..., :N, :N]

        # ===== 新增 2: 距离编码为 attn_bias =====
        # 在最后一维加个 channel, 然后 MLP
        dist_in = pair_dist.unsqueeze(-1)
        # 获取真实节点数（包含 super-node）
        N_full = atom.size(1)  # [B, N+1]
        pad_len = N_full - pair_dist.size(-1)  # 通常为 1，用于 pad super-node

        # ===== 距离编码为 attn_bias =====
        dist_in = pair_dist.unsqueeze(-1)  # [B, N, N, 1]
        bias = self.edge_dist_mlp(dist_in).squeeze(-1)  # [B, N, N]
        bias = F.pad(bias, (pad_len, 0, pad_len, 0))  # → [B, N+1, N+1]
        bias = bias.unsqueeze(1)  # → [B, 1, N+1, N+1]

        # ===== 构造 dist_embed 作为边特征用于 attention =====
        dist_feat = self.edge_dist_mlp(dist_in)  # [B, N, N, d_k]
        dist_feat = F.pad(
            dist_feat, (0, 0, pad_len, 0, pad_len, 0)
        )  # → [B, N+1, N+1, d_k]
        dist_embed = dist_feat.unsqueeze(1).expand(
            -1, self.num_heads, -1, -1, -1
        )  # [B, H, N+1, N+1, d_k]

        # —— 逐层 Message-Passing ——
        for layer in self.EncoderAtomList:
            atom = layer(
                x=atom,
                attn_mask=attn_mask4d[..., :Np1, :Np1],
                dist=pair_dist,
                dist_bar=self.dist_bar,
                attn_bias=bias,  # 把距离 bias 传进去
                dist_embed=dist_embed,
            )

        graph_feature = self.head_Graph(atom[:, 0, :])

        return graph_feature


"""
dist_bar = [0.001]
[SCAGE] dist_bar used = [0.0010000000474974513]
[DEBUG] dist_bar used in attention: [0.0010000000474974513]


dist_bar = [0.01]
[SCAGE] dist_bar used = [0.009999999776482582]
[DEBUG] dist_bar used in attention: [0.009999999776482582]

dist_bar = [0.1]
[SCAGE] dist_bar used = [0.10000000149011612]
[DEBUG] dist_bar used in attention: [0.10000000149011612]
"""
