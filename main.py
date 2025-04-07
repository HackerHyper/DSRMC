from torch import true_divide
from train_test import train
import numpy as np

import random
import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#主函数，设置数据集、模型训练
if __name__ == "__main__":    
    # data_folder = 'BRCA'
    # data_folder = 'ROSMAP'
    data_folder = 'GBM'
    testonly = False
    modelpath = './model/'
    seed = 1995
    print(seed)
    seed_everything(seed)
    train(data_folder, modelpath, testonly)
