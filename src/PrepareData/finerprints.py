import torch
import numpy as np

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from ProcessorData.token_dicts import smartsPatts

def InitKeys(keyList, keyDict):
    """ *Internal Use Only*
    generates SMARTS patterns for the keys, run once
    """
    assert len(keyList) == len(keyDict.keys()), 'length mismatch'
    for key in keyDict.keys():
        patt, count = keyDict[key]
        if patt != '?':
            sma = Chem.MolFromSmarts(patt)
            if not sma:
                print('SMARTS parser error for key #%d: %s' % (key, patt))
            else:
                keyList[key - 1] = sma, count

def count_ring_types(mol, ring_source, condition_fn, base_bit_indices, bits):
    ring_data = getattr(mol.GetRingInfo(), ring_source)()
    temp = {i: 0 for i in range(3, 11)}

    for ring in ring_data:
        if condition_fn(mol, ring):
            ring_len = len(ring)
            if ring_len in temp:
                temp[ring_len] += 1

    for size, base in base_bit_indices.items():
        count = temp[size]
        for i in range(min(count, 5)):
            bits[base + i * 7] = 1

    return bits

def count_ring_types(mol, ring_source, condition_fn, base_bit_indices, bits):
    ring_data = getattr(mol.GetRingInfo(), ring_source)()
    temp = {i: 0 for i in range(3, 11)}

    for ring in ring_data:
        if condition_fn(mol, ring):
            ring_len = len(ring)
            if ring_len in temp:
                temp[ring_len] += 1

    for size, base in base_bit_indices.items():
        count = temp[size]
        for i in range(min(count, 5)):
            bits[base + i * 7] = 1

    return bits

def condition_func1(mol, ring):
    return True

def condition_func2(mol, ring):
    is_saturated = all(mol.GetBondWithIdx(idx).GetBondType().name == 'SINGLE' for idx in ring)
    is_aromatic = all(mol.GetBondWithIdx(idx).GetBondType().name == 'AROMATIC' for idx in ring)
    all_carbon = all(mol.GetBondWithIdx(idx).GetBeginAtom().GetAtomicNum() == 6 and
                     mol.GetBondWithIdx(idx).GetEndAtom().GetAtomicNum() == 6 for idx in ring)
    return is_saturated or (is_aromatic and all_carbon)

def condition_func3(mol, ring):
    is_saturated = all(mol.GetBondWithIdx(idx).GetBondType().name == 'SINGLE' for idx in ring)
    if is_saturated:
        return True
    is_aromatic = all(mol.GetBondWithIdx(idx).GetBondType().name == 'AROMATIC' for idx in ring)
    contain_N = any(mol.GetBondWithIdx(idx).GetBeginAtom().GetAtomicNum() == 7 or
                    mol.GetBondWithIdx(idx).GetEndAtom().GetAtomicNum() == 7 for idx in ring)
    return is_aromatic and contain_N

def condition_func4(mol, ring):
    is_saturated = all(mol.GetBondWithIdx(idx).GetBondType().name == 'SINGLE' for idx in ring)
    if is_saturated:
        return True
    is_aromatic = all(mol.GetBondWithIdx(idx).GetBondType().name == 'AROMATIC' for idx in ring)
    has_hetero = any(mol.GetBondWithIdx(idx).GetBeginAtom().GetAtomicNum() not in [1,6] or
                     mol.GetBondWithIdx(idx).GetEndAtom().GetAtomicNum() not in [1,6] for idx in ring)
    return is_aromatic and has_hetero

def condition_func5(mol, ring):
    is_unsaturated = any(mol.GetBondWithIdx(idx).GetBondType().name != 'SINGLE' for idx in ring)
    is_non_aromatic = all(mol.GetBondWithIdx(idx).GetBondType().name != 'AROMATIC' for idx in ring)
    all_carbon = all(mol.GetBondWithIdx(idx).GetBeginAtom().GetAtomicNum() == 6 and
                     mol.GetBondWithIdx(idx).GetEndAtom().GetAtomicNum() == 6 for idx in ring)
    return is_unsaturated and is_non_aromatic and all_carbon

def condition_func6(mol, ring):
    is_unsaturated = any(mol.GetBondWithIdx(idx).GetBondType().name != 'SINGLE' for idx in ring)
    is_non_aromatic = all(mol.GetBondWithIdx(idx).GetBondType().name != 'AROMATIC' for idx in ring)
    contain_N = any(mol.GetBondWithIdx(idx).GetBeginAtom().GetAtomicNum() == 7 or
                    mol.GetBondWithIdx(idx).GetEndAtom().GetAtomicNum() == 7 for idx in ring)
    return is_unsaturated and is_non_aromatic and contain_N

def condition_func7(mol, ring):
    is_unsaturated = any(mol.GetBondWithIdx(idx).GetBondType().name != 'SINGLE' for idx in ring)
    is_non_aromatic = all(mol.GetBondWithIdx(idx).GetBondType().name != 'AROMATIC' for idx in ring)
    has_hetero = any(mol.GetBondWithIdx(idx).GetBeginAtom().GetAtomicNum() not in [1,6] or
                     mol.GetBondWithIdx(idx).GetEndAtom().GetAtomicNum() not in [1,6] for idx in ring)
    return is_unsaturated and is_non_aromatic and has_hetero

def func_8(mol, bits):
    AllRingsBond = mol.GetRingInfo().BondRings()
    temp = {'aromatic': 0, 'heteroatom': 0}
    for ring in AllRingsBond:
        aromatic = all(mol.GetBondWithIdx(idx).GetBondType().name == 'AROMATIC' for idx in ring)
        if aromatic:
            temp['aromatic'] += 1
        heteroatom = any(mol.GetBondWithIdx(idx).GetBeginAtom().GetAtomicNum() not in [1,6] or
                         mol.GetBondWithIdx(idx).GetEndAtom().GetAtomicNum() not in [1,6] for idx in ring)
        if heteroatom:
            temp['heteroatom'] += 1

    if temp['aromatic'] >= 4:
        for i in range(4):
            bits[140 + i * 2] = 1
    elif temp['aromatic'] == 3:
        bits[140] = bits[142] = bits[144] = 1
    elif temp['aromatic'] == 2:
        bits[140] = bits[142] = 1
    elif temp['aromatic'] == 1:
        bits[140] = 1

    if temp['aromatic'] >= 4 and temp['heteroatom'] >= 4:
        for i in range(4):
            bits[141 + i * 2] = 1
    elif temp['aromatic'] == 3 and temp['heteroatom'] == 3:
        bits[141] = bits[143] = bits[145] = 1
    elif temp['aromatic'] == 2 and temp['heteroatom'] == 2:
        bits[141] = bits[143] = 1
    elif temp['aromatic'] == 1 and temp['heteroatom'] == 1:
        bits[141] = 1
    return bits


PubchemKeys = None
def calcPubChemFingerPart1(mol, **kwargs):
    """  Calculate PubChem Fingerprints （1-115; 263-881)
    **Arguments**
     - mol: the molecule to be fingerprinted
     - any extra keyword arguments are ignored
    **Returns**
      a _DataStructs.SparseBitVect_ containing the fingerprint.
    >>> m = Chem.MolFromSmiles('CNO')
    >>> bv = PubChemFingerPart1(m)
    >>> tuple(bv.GetOnBits())
    (24, 68, 69, 71, 93, 94, 102, 124, 131, 139, 151, 158, 160, 161, 164)
    >>> bv = PubChemFingerPart1(Chem.MolFromSmiles('CCC'))
    >>> tuple(bv.GetOnBits())
    (74, 114, 149, 155, 160)
    """
    global PubchemKeys
    if PubchemKeys is None:
        PubchemKeys = [(None, 0)] * len(smartsPatts.keys())
        InitKeys(PubchemKeys, smartsPatts)
    ctor = kwargs.get('ctor', DataStructs.SparseBitVect)
    res = ctor(len(PubchemKeys) + 1)
    for i, (patt, count) in enumerate(PubchemKeys):
        if patt is not None:
            if count == 0:
                res[i + 1] = mol.HasSubstructMatch(patt)
            else:
                matches = mol.GetSubstructMatches(patt)
                if len(matches) > count:
                    res[i + 1] = 1
    return res

def calcPubChemFingerPart2(mol):
    bits = [0] * 148
    mappings = [
        ('AtomRings', condition_func1, {3: 0, 4: 14, 5: 28, 6: 63, 7: 98, 8: 112, 9: 126, 10: 133}),
        ('BondRings', condition_func2, {3: 1, 4: 15, 5: 29, 6: 64, 7: 99, 8: 113, 9: 127, 10: 134}),
        ('BondRings', condition_func3, {3: 2, 4: 16, 5: 30, 6: 65, 7: 100, 8: 114, 9: 128, 10: 135}),
        ('BondRings', condition_func4, {3: 3, 4: 17, 5: 31, 6: 66, 7: 101, 8: 115, 9: 129, 10: 136}),
        ('BondRings', condition_func5, {3: 4, 4: 18, 5: 32, 6: 67, 7: 102, 8: 116, 9: 130, 10: 137}),
        ('BondRings', condition_func6, {3: 5, 4: 19, 5: 33, 6: 68, 7: 103, 8: 117, 9: 131, 10: 138}),
        ('BondRings', condition_func7, {3: 6, 4: 20, 5: 34, 6: 69, 7: 104, 8: 118, 9: 132, 10: 139}),
    ]
    for ring_src, cond_fn, base_map in mappings:
        bits = count_ring_types(mol, ring_src, cond_fn, base_map, bits)
    bits = func_8(mol, bits)
    return bits

def GetPubChemFPs(mol):
    """*Internal Use Only*
    Calculate PubChem Fingerprints
    """
    mol = Chem.AddHs(mol)
    AllBits=[0]*881
    res1=list(calcPubChemFingerPart1(mol).ToBitString())
    for index, item in enumerate(res1[1:116]):
        if item == '1':
            AllBits[index] = 1
    for index2, item2 in enumerate(res1[116:734]):
        if item2 == '1':
            AllBits[index2+115+148] = 1
    res2=calcPubChemFingerPart2(mol)
    for index3, item3 in enumerate(res2):
        if item3==1:
            AllBits[index3+115]=1
    #AllBits = np.array(AllBits, dtype= np.bool)
    AllBits = np.array(AllBits)
    return AllBits

"""用于提取分子的传统指纹特征（Traditional Fingerprints）和Transformer 学习的指纹特征（Transformer Fingerprints）"""
def getTraditionalFingerprintsFeature(mol: Chem.Mol, smiles=None):
    fp2 = []
    fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
    fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
    fp_pubcfp = GetPubChemFPs(mol)
    # print(f'maccs: {torch.tensor(fp_maccs).shape} pubchem: {torch.tensor(fp_pubcfp).shape} phaerg: {torch.tensor(fp_phaErGfp).shape}')
    fp2.extend(fp_maccs)
    fp2.extend(fp_phaErGfp)
    fp2.extend(fp_pubcfp)
    fp2 = torch.tensor(fp2, dtype=torch.float32)

    return fp2

# 批处理包装器
def batch_traditional_fps(smiles_list):
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Invalid SMILES: {smi}")
            continue
        fp = getTraditionalFingerprintsFeature(mol)
        features.append(fp)
    return torch.stack(features)  # shape: [N, D]

def get_fingerprint(smi):
    """
    处理单个 SMILES 字符串，返回其传统分子指纹向量（Torch Tensor）
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    fp = getTraditionalFingerprintsFeature(mol)
    return fp  # 返回值已为 Tensor 类型