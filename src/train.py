import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from Model.model import *
from ProcessorData.const import *
from torch.utils.data import DataLoader
from torch_geometric.utils import degree
from ProcessorData.dataset import MyTestDataset


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    print("Training on {} samples...".format(len(drug1_loader_train.dataset)))
    model.train()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    total_loss = 0.0

    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data1.y.view(-1, 1).long().to(device)
        y = y.squeeze(1)

        optimizer.zero_grad()
        output = model(data1, data2)
        loss = loss_fn(output, y)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        # softmax outputs predicted scores and predicted labels
        ys = F.softmax(output, 1).to("cpu").data.numpy()
        predicted_labels = list(map(lambda x: np.argmax(x), ys))
        predicted_scores = list(map(lambda x: x[1], ys))
        total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
        total_prelabels = torch.cat(
            (total_prelabels, torch.Tensor(predicted_labels)), 0
        )
        total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)

        if batch_idx % LOG_INTERVAL == 0:
            print(
                "Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data1.y),
                    len(drug1_loader_train.dataset),
                    100.0 * batch_idx / len(drug1_loader_train),
                    loss.item(),
                )
            )
    
    avg_train_loss = total_loss / len(drug1_loader_train)
    return avg_train_loss

def predicting(model, device, drug1_loader_test, drug2_loader_test, loss_fn=None):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    total_loss = 0.0

    print("Make prediction for {} samples...".format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0].to(device)
            data2 = data[1].to(device)
            output = model(data1, data2)
            
            # 计算 loss（如果提供了 loss_fn）
            if loss_fn is not None:
                y = data1.y.view(-1).long().to(device)
                # print("output.shape:", output.shape)
                # print("y.shape:", y.shape)
                loss = loss_fn(output, y)
                total_loss += loss.item()

            ys = F.softmax(output, 1).to("cpu").data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)

    avg_test_loss = total_loss / len(drug1_loader_test) if loss_fn is not None else None
    return (
        total_labels.numpy().flatten(),
        total_preds.numpy().flatten(),
        total_prelabels.numpy().flatten(),
        avg_test_loss
    )


modeling = MultiSyn

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.00001 # 0.0001
LOG_INTERVAL = 20
NUM_EPOCHS = 200

print("Learning rate: ", LR)
print("Epochs: ", NUM_EPOCHS)
datafile = "drugcom_12415"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("The code uses GPU...")
else:
    device = torch.device("cpu")
    print("The code uses CPU!!!")

drug1_data = MyTestDataset(root=DATAS_DIR, dataset=datafile + "_drug1")
drug2_data = MyTestDataset(root=DATAS_DIR, dataset=datafile + "_drug2")
lenth = len(drug1_data)
pot = int(lenth / 5)
print("lenth", lenth)
print("pot", pot)

# 读取固定索引
# fold_data = np.load(INDEX_DIR, allow_pickle=True).item()
# train_idx_list = fold_data["train_idx"]
# test_idx_list = fold_data["test_idx"]

# 5-fold random split
random_num = random.sample(range(0, lenth), lenth)
for i in range(5):
    # Construct DataLoader for training and test sets
    test_num = random_num[pot * i : pot * (i + 1)]
    train_num = random_num[: pot * i] + random_num[pot * (i + 1) :]

    # train_num = train_idx_list[i]
    # test_num = test_idx_list[i]

    drug1_data_train = drug1_data[train_num]
    drug1_data_test = drug1_data[test_num]
    drug1_loader_train = DataLoader(
        drug1_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None, collate_fn=custom_collate
    )
    drug1_loader_test = DataLoader(
        drug1_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None, collate_fn=custom_collate
    )

    drug2_data_test = drug2_data[test_num]
    drug2_data_train = drug2_data[train_num]
    drug2_loader_train = DataLoader(
        drug2_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None, collate_fn=custom_collate
    )
    drug2_loader_test = DataLoader(
        drug2_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None, collate_fn=custom_collate
    )

    # model = modeling(deg=deg).to(device)
    model = modeling().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # if os.path.exists(OPTIMIZER_DIR):
    #     optimizer.load_state_dict(torch.load(OPTIMIZER_DIR))
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_AUCs_test = "fold_" + str(i) + ".csv"
    file_AUCs_test = os.path.join(RESULTS_DIR, file_AUCs_test)
    AUCs = "Epoch,AUC_dev,PR_AUC,ACC,BACC,PREC,TPR,KAPPA,TIME,TRAIN,TEST"
    with open(file_AUCs_test, "w") as f:
        f.write(AUCs + "\n")

    best_auc_train = 0
    best_auc_test = 0
    
    best_loss = float('inf')
    best_auc = 0
    patience = 20
    counter = 0
    
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        train_loss = train(
            model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1
        )
        T_p, S_p, Y_p, test_loss = predicting(model, device, drug1_loader_test, drug2_loader_test, loss_fn=loss_fn)
        # T is correct label
        # S is predict score
        # Y is predict label
        elapsed_time = time.time() - start_time
        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        best_auc_test, auc_test = compute_preformence(
            T_p, S_p, Y_p, best_auc_test, file_AUCs_test, epoch + 1, elapsed_time_str, train_loss, test_loss
        )

        # early stop
        # if test_loss < best_loss or (test_loss == best_loss and auc_test > best_auc):
        #     best_loss = test_loss
        #     best_auc = auc_test
        #     counter = 0  # reset counter
        #     torch.save(model.state_dict(), MODEL_DIR)
        #     torch.save(optimizer.state_dict(), OPTIMIZER_DIR)
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print(f"Early stopping at epoch {epoch}")
        #         break

    break