import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self,
                 num_heads,
                 hidden_dim,
                 dropout=0.1,
                 attn_dropout=0.1,
                 temperature=1,
                 use_super_node=True,
                 knn_ratio: float = 0.3  # 保留最近 α 比例 的邻居
                 ):
        super().__init__()
        self.d_k = hidden_dim // num_heads
        self.num_heads = num_heads
        self.use_super_node = use_super_node
        self.knn_ratio = knn_ratio

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 注意：后续 concat 后的维度从 num_heads * d_k
        self.scale_linear = nn.Sequential(
            nn.Linear(num_heads * self.d_k, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x, dist, mask=None, attn_bias=None, dist_embed=None):
        B, N_full, _ = x.shape
        H, d_k = self.num_heads, self.d_k

        # — QKV —
        q = self.q_proj(x).view(B, N_full, H, d_k).transpose(1,2)  # [B,H,N,d_k]
        k = self.k_proj(x).view(B, N_full, H, d_k).transpose(1,2)
        v = self.v_proj(x).view(B, N_full, H, d_k).transpose(1,2)

        # — 计算原始 scores —
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(d_k)  # [B,H,N,N]
        if attn_bias is not None:
            scores = scores + attn_bias  # 假如你还想保留 bias

        # — 构造带 super-node 的距离矩阵 new_dist —
        d2 = dist.unsqueeze(1)  # [B,1,N,N]
        n_node = dist.size(-1) + int(self.use_super_node)
        new_dist = d2.new_ones([B,1,n_node,n_node])
        new_dist[:, :, int(self.use_super_node):, int(self.use_super_node):] = d2
        # super-node 全开不参与屏蔽
        new_dist[:, :, 0, :] = 0
        new_dist[:, :, :, 0] = 0
        d2 = new_dist[:,0]  # [B, N_full, N_full]

        # — 按比例 KNN 掩码 —
        k_nn = max(1, int(self.knn_ratio * N_full))
        _, idx = torch.topk(d2, k=k_nn, dim=-1, largest=False)  # [B,N_full,k]
        mask_knn = torch.zeros_like(d2, dtype=torch.bool)
        batch_idx = torch.arange(B, device=x.device)[:,None,None]
        node_idx  = torch.arange(N_full, device=x.device)[None,:,None]
        mask_knn[batch_idx, node_idx, idx] = True
        mask_knn[:,0,:] = True
        mask_knn[:,:,0] = True
        attn_mask = mask_knn.unsqueeze(1)  # [B,1,N_full,N_full]

        # — 应用掩码 + softmax —
        scores = scores.unsqueeze(2)       # [B,H,1,N,N]
        scores = scores.masked_fill(~attn_mask.unsqueeze(1), -1e12)
        attn   = F.softmax(scores, dim=-1) # [B,H,1,N,N]
        attn   = self.attn_dropout(attn)

        # — 消息传递 —
        msg = torch.matmul(attn, v.unsqueeze(2))  # [B,H,1,N,d_k]
        msg = msg.squeeze(2)                      # [B,H,N,d_k]

        # — 输出拼接 & 残差 —
        out = msg.transpose(1,2).contiguous().view(B, N_full, H*d_k)
        out = self.scale_linear(out) + x
        out = self.layer_norm(out)
        return out, attn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, ffn_hidden_dim, activation_fn="GELU", dropout=DEFAULTS.DROP_RATE):
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
    def __init__(self, num_heads, hidden_dim, ffn_hidden_dim, dropout=0.1, attn_dropout=0.1, temperature=1,
                 activation_fn='GELU'):
        super(MultiScaleTransformer, self).__init__()
        assert hidden_dim % num_heads == 0
        self.self_attention = MultiScaleAttention(num_heads, hidden_dim, dropout, attn_dropout, temperature)
        self.self_ffn_layer = PositionWiseFeedForward(hidden_dim, ffn_hidden_dim, activation_fn=activation_fn)


    def forward(self, x, dist, attn_mask, attn_bias=None, dist_embed=None):
        x, attn = self.self_attention(x, dist, mask=attn_mask, attn_bias=attn_bias, dist_embed=dist_embed)
        x = self.self_ffn_layer(x)

        return x, attn

class EncoderAtomLayer(nn.Module):
    def __init__(self, hidden_dim, ffn_hidden_dim, num_heads, dropout=DEFAULTS.DROP_RATE,
                 attn_dropout=DEFAULTS.DROP_RATE, temperature=1, activation_fn='GELU'):
        super(EncoderAtomLayer, self).__init__()
        self.transformer = MultiScaleTransformer(num_heads, hidden_dim, ffn_hidden_dim, dropout, attn_dropout,
                                                 temperature, activation_fn)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer_norm.weight)

    def forward(self, x, attn_mask: Tensor,
                attn_bias: Tensor=None, dist=None, dist_embed=None):

        x, attn = self.transformer(x, dist, attn_mask, attn_bias, dist_embed)

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

def clean_fg_dict_with_max_coverage_fn_group_priority(fg_dict: Dict[int, List[List[int]]]) \
        -> Dict[int, List[List[int]]]:
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
    fg_dict = {fn_id: [match for match in matches if match] for fn_id, matches in fg_dict.items()}
    return fg_dict

def get_fg_list(n_atom: int, fg_dict: Dict[int, List[List[int]]]) -> Tuple[List[int], List[int]]:
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
    edges_index = [0 if fg_index_arr[edges[i][0]] == fg_index_arr[edges[i][1]] else 1 for i in range(len(edges))]
    return fg, edges_index


@torch.jit.script
def gaussian(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    # noinspection PyTypeChecker
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (torch.sqrt(2 * torch.pi) * std)

# noinspection PyPep8Naming
class GaussianKernel(nn.Module):
    def __init__(self, K: int = 128, std_width: float = 1.0, start: float = 0.0, stop: float = 9.0):
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
            embed = nn.Embedding(CompoundKit.get_atom_feature_size(name) + 5, embed_dim, padding_idx=0)
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
            nn.Linear(num_kernel, embed_dim)
        )

        self.van_der_waals_radis_embedding = nn.Sequential(
            GaussianKernel(K=num_kernel, std_width=1.0, start=0.0, stop=9.0),
            nn.Linear(num_kernel, embed_dim)
        )

        self.partial_charge_embedding = nn.Sequential(
            GaussianKernel(K=num_kernel, std_width=1.0, start=0.0, stop=9.0),
            nn.Linear(num_kernel, embed_dim)
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
            van_der_waals_radis_embed = self.van_der_waals_radis_embedding(node_features["van_der_waals_radius"])
            out_embed += van_der_waals_radis_embed
        if "partial_charge" in node_features:
            partial_charge_embed = self.partial_charge_embedding(node_features["partial_charge"])
            out_embed += partial_charge_embed

        graph_token_embed = self.graph_embedding.weight.unsqueeze(0).repeat(out_embed.size()[0], 1, 1)

        out_embed = torch.cat([graph_token_embed, out_embed], dim=1)
        # normalize
        # out_embed = out_embed / (out_embed.norm(dim=-1, keepdim=True) + 1e-5)
        out_embed = self.final_layer_norm(out_embed)
        return out_embed

# noinspection PyPep8Naming,SpellCheckingInspection
class Scage(nn.Module):
    def __init__(self, mode, atom_names, atom_embed_dim, num_kernel,
                 layer_num, num_heads, hidden_size, graph_output_dim=512):
        super(Scage, self).__init__()

        self.mode = mode
        self.atom_names = atom_names
        self.atom_embed_dim = atom_embed_dim
        self.num_kernel = num_kernel
        self.layer_num = layer_num
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.graph_output_dim = graph_output_dim

        self.register_buffer('dist_bar', torch.tensor(GlobalVar.dist_bar))
        
        D = atom_embed_dim  # e.g. 512

        # —— 新增 —— 坐标编码 MLP —— 把每个原子 (x,y,z) → D 维向量
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )

        # —— 新增 —— 距离编码 MLP —— 把标量距离 → 一个 attention bias
        # 输出一个标量 bias（也可以输出多头维度再 reshape）
        self.edge_dist_mlp = nn.Sequential(
            nn.Linear(1, D//num_heads),   # 或者 1,1，根据你想插入 attn_bias 的方式决定
            nn.ReLU(),
            nn.Linear(D//num_heads, 1)
        )
        
        self.atom_feature = AtomEmbedding(self.atom_names, self.atom_embed_dim, self.num_kernel)

        self.EncoderAtomList = nn.ModuleList()
        for j in range(self.layer_num):
            self.EncoderAtomList.append(
                EncoderAtomLayer(self.atom_embed_dim, self.hidden_size, self.num_heads)
            )

        self.head_Graph = AtomProjection(self.atom_embed_dim, self.atom_embed_dim // 2, self.graph_output_dim)

    def forward(self, batched_data):
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
            "atomic_num":       atom_feats[:,:,0].long(),
            "chiral_tag":       atom_feats[:,:,1].long(),
            "is_aromatic":      atom_feats[:,:,2].long(),
            "degree":           atom_feats[:,:,3].long(),
            "explicit_valence": atom_feats[:,:,4].long(),
            "formal_charge":    atom_feats[:,:,5].long(),
            "num_explicit_Hs":  atom_feats[:,:,6].long(),
            "hybridization":    atom_feats[:,:,7].long(),
            "total_numHs":      atom_feats[:,:,8].long(),
            "atom_is_in_ring":  atom_feats[:,:,9].long(),
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
        mask2d = torch.cat([super_mask, node_mask], dim=1)        # [B, N_full]

        # 构造 pairwise mask：只有两端节点都存在的对才有效
        # mask2d[:, :, None]: [B, N_full, 1], mask2d[:, None, :]: [B, 1, N_full]
        pair_mask = mask2d.unsqueeze(2) & mask2d.unsqueeze(1)     # [B, N_full, N_full]

        # 现在才是真正的 attention mask
        attn_mask4d = pair_mask.unsqueeze(1)                     # [B,1,N_full,N_full]    

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
        dist_in = pair_dist.unsqueeze(-1)                     # [B, N, N, 1]
        bias = self.edge_dist_mlp(dist_in).squeeze(-1)        # [B, N, N]
        bias = F.pad(bias, (pad_len, 0, pad_len, 0))          # → [B, N+1, N+1]
        bias = bias.unsqueeze(1)                              # → [B, 1, N+1, N+1]

        # ===== 构造 dist_embed 作为边特征用于 attention =====
        dist_feat = self.edge_dist_mlp(dist_in)               # [B, N, N, d_k]
        dist_feat = F.pad(dist_feat, (0, 0, pad_len, 0, pad_len, 0))  # → [B, N+1, N+1, d_k]
        dist_embed = dist_feat.unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)  # [B, H, N+1, N+1, d_k]
        
        # —— 逐层 Message-Passing —— 
        for layer in self.EncoderAtomList:
            atom = layer(
                x         = atom, 
                attn_mask = attn_mask4d[..., :Np1, :Np1],
                dist      = pair_dist, 
                # dist_bar  = self.dist_bar,
                attn_bias = bias,      # 把距离 bias 传进去
                dist_embed = dist_embed   
            )

        graph_feature  = self.head_Graph(atom[:, 0, :])

        return graph_feature