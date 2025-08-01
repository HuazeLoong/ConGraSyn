# ConGraSyn: A Conformation Enhanced Graph Attention Framework for Predicting Synergistic Drug Combinations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16215274.svg)](https://doi.org/10.5281/zenodo.16215274)

This repository contains the official implementation of our paper:  
**ConGraSyn** is a scale conformation-aware framework for predicting synergistic drug combinations by integrating 3D-enhanced molecular graphs with PPI-informed and drug-induced cell line features.

<!-- ![ConGraSyn Architecture](ConGraSyn.png) -->

<!-- You can find full documentation here:  [https://HuazeLoong.github.io/ConGraSyn/](https://HuazeLoong.github.io/ConGraSyn/) -->

## 1. Introduction

ConGraSyn represents molecules as heterogeneous molecular graphs and predicts drug combination synergy using graph neural networks.  
It provides substructure-level attention and integrates multi-source data, including PPI and cell lines omics profiles.

**Paper Link**: *Coming soon...*

## 1.1 Features
This method explicitly embeds 3D atomic coordinates and interatomic distances in 2D molecular graphs, introduces a scale conformational learning (MCL) module to capture local and global structural semantics, and fuses with molecular fingerprints to supplement global structural information. At the same time, the pre-trained TranSiGen model is used to generate drug-induced transcriptome features, which are fused with baseline omics data to characterize the dynamic response of cells to drugs.


## 1.2 File Structure

```text
ConGraSyn/             ← Project root directory
├── setup.py          ← Packaging and installation configuration
├── requirements.txt  ← Dependency management
├── README.md         ← Project description
└── src/
    └── ConGraSyn/         ← Python package (contains all core source code)
        ├── __init__.py
        ├── Datas           ← Data folder
        ├── Model           ← Model script file
        ├── PrepareData      ← Data preprocessing script file
        ├── ProcessorData   ← Data processing script file
            └── const.py
        ├── config.py
        ├── train.py
        └── utils.py
```

## 1.3 Citation
If you find this repository helpful, please cite our work:

```bibtex

```

# 2. Usage
## 2.1 Requirements
We recommend the following Python environment:
```bash
# ---- Core Deep Learning Framework ----
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0

# ⚠ torch-scatter must match your PyTorch and CUDA version.
# Manual installation is recommended (see notes below).

# ---- GNN Packages ----
torch-geometric==2.4.0
dgl==1.1.2  # or dgl==1.1.2+cu118 depending on your CUDA version

# ---- Chemistry Toolkit ----
rdkit==2022.9.5  # from conda or RDKit wheels

# ---- ML + Data Processing ----
scikit-learn>=1.2.0
numpy>=1.24.0
pandas>=1.3.0
scipy>=1.7.0

# ---- Optional Utilities ----
tqdm
matplotlib
```

Install core dependencies using:

```bash
pip install -r requirements.txt
```

**Notes on Specific Dependencies**

⚠ torch-scatter
torch-scatter requires a PyTorch- and CUDA-matching build. Use the following command to install a compatible version:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```
You can find more options at: [PyG Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

⚠ rdkit
rdkit is not available on PyPI; it is recommended to install via conda:
```bash
conda install -c rdkit rdkit==2022.9.5
```

## 2.2 Preprocessing
Before cloning the code, please download the data in our Datas folder at https://doi.org/10.5281/zenodo.16210069 and put it into Datas. You can also download the full code here.

To preprocess the drug combination dataset:

```bash
python prepare_data.py
```

Processed files will be saved to `ConGraSyn\datas\processed`.

## 2.3 Train the Model
To train the ConGraSyn model:
```bash
python train.py
```
Results will be saved to the `ConGraSyn\datas\results` directory.
