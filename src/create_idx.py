import os
import torch
import random
import numpy as np

from utils import *
from Model.model import *
from ProcessorData.const import *
from ProcessorData.dataset import MyTestDataset


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


modeling = MultiSyn

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.0001
LOG_INTERVAL = 20
NUM_EPOCHS = 400

print("Learning rate: ", LR)
print("Epochs: ", NUM_EPOCHS)
datafile = "drugcom_12415"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("The code uses GPU...")
else:
    device = torch.device("CPU")
    print("The code uses CPU!!!")

drug1_data = MyTestDataset(root=DATAS_DIR, dataset=datafile + "_drug1")
drug2_data = MyTestDataset(root=DATAS_DIR, dataset=datafile + "_drug2")
lenth = len(drug1_data)
pot = int(lenth / 5)
print("lenth", lenth)
print("pot", pot)

train_idx_list = []
test_idx_list = []

# 5-fold random split
random_num = random.sample(range(0, lenth), lenth)
for i in range(5):
    # Construct DataLoader for training and test sets
    test_idx = random_num[pot * i : pot * (i + 1)]
    train_idx = random_num[: pot * i] + random_num[pot * (i + 1) :]
    test_idx_list.append(test_idx)
    train_idx_list.append(train_idx)

fold_dict = {
    "train_idx": train_idx_list,
    "test_idx": test_idx_list
}

os.makedirs("fold_indices", exist_ok=True)
np.save("fold_indices/fold_train_test_indices.npy", fold_dict)
print("五折 train/test 索引保存完成: fold_indices/fold_train_test_indices.npy")
