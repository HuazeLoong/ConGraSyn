import os

# # 获取当前文件所在目录的绝对路径
# SUB_PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

# 修改为项目根目录 Testsyn
SUB_PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 数据根目录
DATAS_DIR = os.path.join(SUB_PROJ_DIR, "Datas")

# 存放实验结果的目录
RESULTS_DIR = os.path.join(DATAS_DIR, "results")

# 存放覆盖范围的目录
# COVERAGE_DIR = os.path.join(DATAS_DIR, "results")

# 存放所有输入数据的目录
DATA_DIR = os.path.join(DATAS_DIR, "data")

# 存放所有扰动数据的目录
PERTUR_DIR = os.path.join(DATA_DIR, "prediction_perturbation")

# 药物ID映射文件（csv格式）
DRUGID_DIR = os.path.join(DATA_DIR, "drug_id.csv")

# 协同药物组合文件
DRUGCOMS_DIR = os.path.join(DATA_DIR, "drug_com.csv")

# 细胞系特征文件(融合PPI)
CELL_FEA_DIR = os.path.join(DATA_DIR, "cell_feat.npy")

# 细胞系编号文件
CELL_ID_DIR = os.path.join(DATA_DIR, "cell2id.tsv")

# 原始细胞系特征
CELL_DIR = os.path.join(DATA_DIR, "cell_features.csv")

# 药物的 SMILES 或结构信息
MOL_DIR = os.path.join(DATA_DIR, "mol.csv")

NEW_FG_GROUPS_DIR = os.path.join(DATA_DIR, "new_fg_groups.txt")

# FILE_AUCS_TEST = os.path.join(RESULT_DIR,'test')

# 数据索引
INDEX_DIR = os.path.join(DATA_DIR, "fold_indices.npy")

# 模型参数
MODEL_DIR = os.path.join(RESULTS_DIR, "best_model.pth")

# 优化器
OPTIMIZER_DIR = os.path.join(RESULTS_DIR, "opimizer.pth")
