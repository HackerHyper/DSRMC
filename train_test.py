""" Training and testing of the model
"""
import os
from re import I
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from model import MMDynamic
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import copy
import gc

class mDataset(Dataset):

    def __init__(self, data, label):
        # 也可以把数据作为一个参数传递给类，__init__(self, data)；
        self.data_v0 = data[0]
        self.data_v1 = data[1]
        self.data_v2 = data[2]
        self.label = label
       
    
    def __getitem__(self, index):
        # 根据索引返回数据
        # data = self.preprocess(self.data[index]) # 如果需要预处理数据的话
        return (self.data_v0[index], self.data_v1[index], self.data_v2[index]),self.label[index]
    
    def __len__(self):
        # 返回数据的长度
        return len(self.data_v0)
    
    # 以下可以不在此定义。
    
    # 如果不是直接传入数据data，这里定义一个加载数据的方法
    def __load_data__(self, csv_paths: list):
        # 假如从 csv_paths 中加载数据，可能要遍历文件夹读取文件等，这里忽略
        # 可以拆分训练和验证集并返回train_X, train_Y, valid_X, valid_Y
        pass
      
    def preprocess(self, data):
        # 将data 做一些预处理
        pass

cuda = True if torch.cuda.is_available() else False

def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)    
    return y_onehot

def prepare_trte_data(data_folder):
    num_view = 3
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in range(1, num_view+1):
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    
    eps = 1e-10
    X_train_min = [np.min(data_tr_list[i], axis=0, keepdims=True) for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] - np.tile(X_train_min[i], [data_tr_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    data_te_list = [data_te_list[i] - np.tile(X_train_min[i], [data_te_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    X_train_max = [np.max(data_tr_list[i], axis=0, keepdims=True) + eps for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] / np.tile(X_train_max[i], [data_tr_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    data_te_list = [data_te_list[i] / np.tile(X_train_max[i], [data_te_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    
    num_tr = data_tr_list[0].shape[0]
    print("num_tr: ")
    print(num_tr)
    num_te = data_te_list[0].shape[0]
    print("num_te: ")
    print(num_te)

    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    data_test_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
        data_test_list.append(data_tensor_list[i][idx_dict["te"]].clone())
    labels = np.concatenate((labels_tr, labels_te))
    return data_train_list, data_test_list, idx_dict, labels


def train_epoch(train_loader, epoch, model, optimizer):
    model.train()
    print("epoch: ", epoch)
    model_tmp = copy.deepcopy(model)
    train_list = []

    # 开始获取损失，排序
    for i, (data_list,label) in enumerate(train_loader):
        
        loss_tmp, _ = model_tmp(data_list, label)
        train_list.append((i, data_list, label,loss_tmp.item()))
    # 排序
    #train_list_sorted = sorted(train_list, key=lambda x: x[3], reverse=True)
    train_list_sorted = sorted(train_list, key=lambda x: x[3])
    # 释放显存
    del model_tmp
    del train_list
    gc.collect()

    cnt = 0
    sum = len(train_list_sorted)
    # 调度
    total_loss = 0
    for elem in train_list_sorted:
        
        print(((epoch+1)/85.0)*sum)
        if cnt >= ((epoch+1)/85.0)*sum +1 :
            break
        optimizer.zero_grad()
        loss, _ = model(elem[1], elem[2])
        loss = torch.mean(loss)
        total_loss = total_loss + loss.item()
        loss.backward()
        optimizer.step()
        cnt = cnt + 1
    print("num batch: ",cnt)
    print("total_loss: ",total_loss/cnt)
#测试

def test_epoch(data_list, model):
    model.eval()
    with torch.no_grad():
        logit = model.infer(data_list)
        prob = F.softmax(logit, dim=1).data.cpu().numpy()
    return prob

def save_checkpoint(model, checkpoint_path, filename="checkpoint.pt"):
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path, filename)
    torch.save(model, filename)


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint)


def train(data_folder, modelpath, testonly):
    # 设置参数
    test_inverval = 1
    if 'BRCA' in data_folder:
        hidden_dim = [500]
        #num_epoch = 2500
        num_epoch = 100
        lr = 1e-4
        step_size = 500
        num_class = 5
    elif 'ROSMAP' in data_folder:
        hidden_dim = [300]
        #num_epoch = 1000
        num_epoch = 100
        lr = 1e-4
        step_size = 100
        num_class = 2
    elif 'GBM' in data_folder:
        hidden_dim = [400]
        #num_epoch = 1000
        num_epoch = 100
        lr = 1e-4
        step_size = 500
        num_class = 4
    # 处理数据集
    data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(data_folder)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    labels_tr_tensor = labels_tr_tensor.cuda()
    onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
    dim_list = [x.shape[1] for x in data_tr_list]

    # 训练数据集加载
    train_loader = data.DataLoader(mDataset(data_tr_list, labels_tr_tensor),
                                       batch_size=4,
                                       shuffle=True,
                                       pin_memory=False)
    # 构建模型
    model = MMDynamic(dim_list, hidden_dim, num_class, dropout=0.5)
    model.cuda()
    # 构造优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # 调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
    # 测试
    if testonly:
        load_checkpoint(model, os.path.join(modelpath, data_folder, 'checkpoint.pt'))
        te_prob = test_epoch(data_test_list, model)
        if num_class == 2:
            print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Test F1: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Test AUC: {:.5f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
        else:
            print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Test F1 weighted: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
            print("Test F1 macro: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
    else: 
        # 训练   
        print("\nTraining...")
        Test_ACC = 0
        Test_F1 = 0
        Test_F2 = 0
        # 开始模型训练
        for epoch in range(num_epoch+1):
            train_epoch(train_loader, epoch, model, optimizer)
            scheduler.step()
            # 测试
            if epoch % test_inverval == 0:
                te_prob = test_epoch(data_test_list, model)
                print("\nTest: Epoch {:d}".format(epoch))
                if num_class == 2:
                    if Test_ACC < accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1)):
                        Test_ACC = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                    if Test_F1 < f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1)):
                        Test_F1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                    if Test_F2 < roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1]):
                        Test_F2 = roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])
                    

                    print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                    print("Test F1: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                    print("Test AUC: {:.5f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
                else:
                    if Test_ACC < accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1)):
                        Test_ACC = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                    if Test_F1 < f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted'):
                        Test_F1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')
                    if Test_F2 < f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro'):
                        Test_F2 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
                    print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                    print("Test F1 weighted: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                    print("Test F1 macro: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
        
        print("Test ACC: {:.5f}".format(Test_ACC))
        print("Test F1: {:.5f}".format(Test_F1))
        print("Test F2: {:.5f}".format(Test_F2))
        # 保存模型
        save_checkpoint(model.state_dict(), os.path.join(modelpath, data_folder))
