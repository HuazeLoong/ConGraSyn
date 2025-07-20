import math
import torch
import numpy as np
import torch.nn as nn

from rdkit import Chem
from torch.autograd import Variable

"""
SMILES-Transformer（编码器+位置编码）
对 SMILES 进行 masked 语言建模预训练
"""
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # (T,H)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)].to(x.device), requires_grad=False)
        return self.dropout(x)

class PreLNTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-LN attention
        src2 = self.self_attn(self.norm1(src), self.norm1(src), self.norm1(src),
                              attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        # Pre-LN feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        src = src + self.dropout2(src2)
        return src

class TrfmSeq2seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super(TrfmSeq2seq, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        # self.trfm = nn.Transformer(d_model=hidden_size, nhead=4,
        #                            num_encoder_layers=n_layers, num_decoder_layers=n_layers,
        #                            dim_feedforward=hidden_size)
        self.encoder_layers = nn.ModuleList([
            PreLNTransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, out_size)

    # def forward(self, src):
    #     # src: (T,B)
    #     embedded = self.embed(src)  # (T,B,H)
    #     embedded = self.pe(embedded)  # (T,B,H)
    #     hidden = self.trfm(embedded, embedded)  # (T,B,H)
    #     out = self.out(hidden)  # (T,B,V)
    #     out = F.log_softmax(out, dim=2)  # (T,B,V)
    #     return out  # (T,B,V)

    # 不再 detach 和 numpy，直接返回 Tensor，保持计算图
    def encode(self, src):
        """
        返回每个分子的分子级嵌入向量，支持 batch。
        输入 src: (T, B) 的 token 序列
        输出: (B, 4H) 分子级别嵌入
        """
        if src.ndim == 1:
            src = src.unsqueeze(1)  # (T, 1)

        src = src.to(self.embed.weight.device)
        embedded = self.embed(src)           # (T,B,H)
        embedded = self.pe(embedded)         # (T,B,H)

        # output = embedded
        # for i in range(self.trfm.encoder.num_layers - 1):
        #     output = self.trfm.encoder.layers[i](output)  # (T,B,H)
        # penul = output.clone()
        # output = self.trfm.encoder.layers[-1](output)
        output = embedded
        for i in range(len(self.encoder_layers) - 1):
            output = self.encoder_layers[i](output)
        penul = output.clone()
        output = self.encoder_layers[-1](output)

        output = self.encoder_norm(output)

        # if self.trfm.encoder.norm:
        #     output = self.trfm.encoder.norm(output)

        # 分子表示拼接：[mean, max, first token, penultimate first token]
        mean_feat = torch.mean(output, dim=0)          # (B,H)
        max_feat = torch.max(output, dim=0).values     # (B,H)
        first_feat = output[0, :, :]                   # (B,H)
        penul_feat = penul[0, :, :]                    # (B,H)
        return torch.cat([mean_feat, max_feat, first_feat, penul_feat], dim=1)  # (B, 4H)

    def encode_one(self, src):
        """
        对单个分子进行encode
        :param src: smiles串
        :return: 1024维指纹
        """
        src = src.unsqueeze(1)
        return self.encode(src)

    def encode_one_all(self, src):
        """
        得到smiles串各个部分的feature
        :param src: string
        :return: len(token) * 256
        """
        src = src.unsqueeze(1)
        src = src.to(self.embed.weight.device)  # 确保输入和 embedding 在同一设备上
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded)  # (T,B,H)
        output = embedded
        # for i in range(self.trfm.encoder.num_layers):
        #     output = self.trfm.encoder.layers[i](output, None)  # (T,B,H)
        for layer in self.encoder_layers:
            output = layer(output)
        output = self.encoder_norm(output)

        # if self.trfm.encoder.norm:
        #     output = self.trfm.encoder.norm(output)  # (T,B,H)
        output = output.detach().numpy()
        return output
    

"""
调用模型 encode 获取分子表示并训练任务
"""
def Get_Atom_Feature(smi, model):
    """
    :param smi: 单个SMILES串
    :return: SMILES串中所有原子的特征，按照SMILES串中出现的顺序 shape = (len(atom_list), 256)
    """
    fea = model.encode_one_all(smi)
    fea = fea.squeeze()

    Mol = Chem.MolFromSmiles(smi)
    atom_list = [atom.GetSymbol() for atom in Mol.GetAtoms()]
    now = 0
    i = 0
    k = 0
    Atom_Fea = np.zeros((len(atom_list), 256), dtype=np.float32)
    while i < len(smi):
        if now < len(atom_list):
            atom = atom_list[now]
            candidate = smi[i:i + len(atom)]

            if candidate.upper() == atom.upper():
                Atom_Fea[now] = fea[k]
                now += 1
                i += len(atom)
                k += 1
                continue

        # 处理特殊符号（顺序不可乱）
        if smi[i:i+2] in ['+1', '-1', '+2', '-2', '+3', '-3', '+4', '-4', '+5', '-5', '+6', '-6', '+7', '-7', '+8', '-8']:
            i += 2
        elif smi[i] == '%' and i + 2 < len(smi):
            i += 3
        elif smi[i:i+2] == '@@':
            i += 2
        else:
            i += 1
        k += 1

    return Atom_Fea

def Get_MolFP(mol, model):
    """
    使用 SMILES-Transformer 提取嵌入特征
    :param mol: token ids，(B, T) 的转置（传入为 (T, B)）
    :param model: TrfmSeq2seq 实例
    :return: (B, 1024) 的张量，参与端到端训练
    """
    return model.encode(torch.t(mol))  # 注意 token 是 (T,B) 格式