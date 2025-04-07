""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x



class MMDynamic(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        #视图个数
        self.views = len(in_dim)

        params = torch.ones(self.views, requires_grad=True)

        self.params = torch.nn.Parameter(params)
        #类别
        self.classes = num_class
        self.dropout = dropout
        
        #循环每一个视图
        # self.FeatureInforEncoder = nn.ModuleList([LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        # #预测单个视图的置信度
        # self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        # #单个视图分类器
        # self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        #单个视图的抽取器
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.Gate = nn.ModuleList([LinearLayer(hidden_dim[0], hidden_dim[0]) for view in range(self.views)])

        self.fusion = nn.Sequential(
            # nn.LayerNorm(hidden_dim[0]),
            nn.Linear(hidden_dim[0],hidden_dim[0]*4),
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.LayerNorm(self.common_dim*4),
            nn.Linear(hidden_dim[0]*4,hidden_dim[0]),
            nn.ReLU(),
            nn.Dropout(0.1)
            )

        
        self.MMClasifier = nn.Linear(hidden_dim[0], num_class)

        # self.MMClasifier = []
        # for layer in range(1, len(hidden_dim)-1):
        #     self.MMClasifier.append(LinearLayer(self.views*hidden_dim[0], hidden_dim[layer]))
        #     self.MMClasifier.append(nn.ReLU())
        #     self.MMClasifier.append(nn.Dropout(p=dropout))
        # if len(self.MMClasifier):
        #     self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        # else:
        #     self.MMClasifier.append(LinearLayer(self.views*hidden_dim[-1], num_class))
        
        # self.MMClasifier = nn.Sequential(*self.MMClasifier)


    def forward(self, data_list, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            Feat = data_list[view]
            #gate
            # FeatureInfo[view] = torch.sigmoid(self.FeatureInforEncoder[view](Feat))
            # feature[view] = data_list[view] * FeatureInfo[view]
            #fc
            feature[view] = self.FeatureEncoder[view](Feat)
            FeatureInfo[view] = torch.sigmoid(self.Gate[view](feature[view]))
            feature[view] = feature[view]* FeatureInfo[view]

            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            # #单个视图的分类器
            # TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            # #预测置信度网络
            # TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])
            # #合并特征和权重
            # feature[view] = feature[view] * TCPConfidence[view]
        #concat各个视图特征
        feat_vec = 0
        for view in range(self.views):
            feat_vec = feat_vec + feature[view]
        fusion_vec = feat_vec
        #分类器
        MMlogit = self.MMClasifier(fusion_vec)
        if infer:
            return MMlogit
        MMLoss = torch.mean(criterion(MMlogit, label))
        # for view in range(self.views):
        #     MMLoss = MMLoss+torch.mean(FeatureInfo[view])
        #     pred = F.softmax(TCPLogit[view], dim=1)
        #     p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
        #     confidence_loss = torch.mean(F.mse_loss(TCPConfidence[view].view(-1), p_target)+criterion(TCPLogit[view], label))
        #     MMLoss = MMLoss+confidence_loss
        return MMLoss, fusion_vec, feature, MMlogit
    
    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit

            


