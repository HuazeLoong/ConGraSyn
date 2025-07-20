import re
import torch
import random
from rdkit import Chem
from ProcessorData.token_dicts import *

def get_inputs(sm):
    seq_len = 220
    sm = sm.split()
    if len(sm) > 218:
        # print('SMILES is too long ({:d})'.format(len(sm)))
        sm = sm[:109] + sm[-109:]
    ids = [NewDic.get(token, NewDic['<unk>']) for token in sm]
    ids = [NewDic['<sos>']] + ids + [NewDic['<eos>']]
    seg = [1] * len(ids)
    padding = [NewDic['<pad>']] * (seq_len - len(ids))
    ids.extend(padding), seg.extend(padding)
    return ids, seg

def get_array(smiles):
    x_id, x_seg = [], []
    for sm in smiles:
        a, b = get_inputs(sm)
        x_id.append(a)
        x_seg.append(b)
    return torch.tensor(x_id)
    # return torch.tensor(x_id), torch.tensor(x_seg)

def smi_tokenizer(smi, max_len, padding=True, mask_prob=0):
    """
    使用 RDKit 解析 SMILES，并进行分词、掩码替换、索引映射与可选 padding。
    """
    smi = smi.strip()
    mol = Chem.MolFromSmiles(smi)
    atom_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    tokens, now, i = [], 0, 0
    while i < len(smi):
        matched = False
        if now < len(atom_list):
            atom = atom_list[now]
            fragment = smi[i:i + len(atom)]
            if fragment.upper() == atom.upper():
                tokens.append(fragment)
                i += len(atom)
                now += 1
                matched = True
        if not matched:
            if smi[i:i + 2] in {'+1', '-1', '+2', '-2', '+3', '-3', '+4', '-4', '+5', '-5', '+6', '-6', '+7', '-7', '+8', '-8'}:
                tokens.append(smi[i:i + 2])
                i += 2
            elif smi[i:i + 3].startswith('%') and len(smi[i:i + 3]) == 3:
                tokens.append(smi[i:i + 3])
                i += 3
            elif smi[i:i + 2] == '@@':
                tokens.append('@@')
                i += 2
            else:
                tokens.append(smi[i])
                i += 1

    # 掩码替换（可选）
    if mask_prob > 0:
        tokens = ['MSK' if random.random() < mask_prob else t for t in tokens]

    # 映射为索引序列
    content = [Token2Idx.get(t, Token2Idx['UNK']) for t in tokens]
    ids = [Token2Idx['BEG']] + content + [Token2Idx['END']]

    # Padding（可选）
    if padding:
        if len(ids) > max_len:
            # 对过长的序列进行截断（前后各保留一半）
            half = (max_len - 2) // 2
            ids = [Token2Idx['BEG']] + content[:half] + content[-half:] + [Token2Idx['END']]
        pad_len = max_len - len(ids)
        ids += [Token2Idx['PAD']] * pad_len

    return ids


def split(sm):
    """
    将 SMILES 字符串拆分成 Token。优先匹配二元原子符号（如 Cl、Br、Na、Si、Se、Mg 等）、
    编号环、手性符号等。
    :param sm: 原始 SMILES 字符串
    :return: 以空格分隔的 token 字符串
    """
    # 优先匹配特殊结构：如 %10, @@, Cl, Br, Na, Si 等，顺序不能错
    pattern = r"%\d{2}|" \
              r"\+\d|" \
              r"-\d|" \
              r"@@|" \
              r"Cl|Br|Si|Se|Na|Mg|Ca|Cu|Be|Ba|Bi|Sr|Ni|" \
              r"Rb|Ra|Xe|Li|Al|As|Ag|Au|Mn|Te|Zn|Fe|Kr|" \
              r"si|se|te|He|" \
              r"."  # 匹配剩余的单字符

    tokens = re.findall(pattern, sm)
    return ' '.join(tokens)