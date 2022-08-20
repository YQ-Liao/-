import os
import urllib  #用于发送网络请求
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import scipy.sparse as sp   #用于处理稀疏矩阵，提供切片索引
from zipfile import ZipFile #用于处理压缩包
from sklearn.model_selection import train_test_split
import pickle #用于把对象二进制序列化和反序列化编码
import pandas as pd
import torch_scatter
import matplotlib.pyplot as plt
from collections import Counter
import scipy.io as scio

#---------定义部分----------

#---数据下载和预处理---

# 获得图和节点的标签和索引，以及图的邻接矩阵
#每个节点都代表一个氨基酸，总共20多种；图的标签只有两种，即是不是酶

class DDDataset(object):

    url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/DD.zip"# 使用D&D数据集

    def __init__(self, data_root="data", train_size=0.8):
        self.data_root = data_root
        self.maybe_download()  # 下载 并解压
        sparse_adjacency, node_labels, graph_indicator, graph_labels = self.read_data()
        # 把coo格式转换为csr 进行稀疏矩阵运算
        self.sparse_adjacency = sparse_adjacency.tocsr()
        self.node_labels = node_labels
        self.graph_indicator = graph_indicator
        self.graph_labels = graph_labels

        self.train_index, self.test_index = self.split_data(train_size)#切割数据集
        self.train_label = graph_labels[self.train_index]  # 得到训练集中所有图对应的类别标签
        self.test_label = graph_labels[self.test_index]  # 得到测试集中所有图对应的类别标签

    def split_data(self, train_size):
        unique_indicator = np.asarray(list(set(self.graph_indicator)))
        # 随机划分训练集和测试集 得到各自对应的图索引   （一个图代表一条数据）
        train_index, test_index = train_test_split(unique_indicator,
                                                   train_size=train_size,
                                                   random_state=1234)
        return train_index, test_index

    def __getitem__(self, index):

        mask = self.graph_indicator == index
        # 得到图索引为index的图对应的所有节点(索引)
        graph_indicator = self.graph_indicator[mask]
        # 每个节点对应的特征标签
        node_labels = self.node_labels[mask]
        # 该图对应的类别标签
        graph_labels = self.graph_labels[index]
        # 该图对应的邻接矩阵
        adjacency = self.sparse_adjacency[mask, :][:, mask]
        return adjacency, node_labels, graph_indicator, graph_labels

    def __len__(self):
        return len(self.graph_labels)
#----------------------------------修改12---------------------------------------------------------------
    def read_data(self):
        # 解压后的路径
        data_dir = os.path.join(self.data_root, "DD")
        print("Loading DD_A.txt")
        # 从txt文件中读取邻接表(每一行可以看作一个坐标，即邻接矩阵中非0值的位置)  包含所有图的节点
        adjacency_list = np.genfromtxt(os.path.join(data_dir, "DD_A.txt"),
                                       dtype=np.int64, delimiter=',') - 1
        print("Loading DD_node_labels.txt")
        # 读取节点的特征标签（包含所有图） 每个节点代表一种氨基酸 氨基酸有20多种，所以每个节点会有一个类型标签 表示是哪一种氨基酸
        node_labels = np.genfromtxt(os.path.join(data_dir, "DD_node_labels.txt"),
                                    dtype=np.int64) - 1      #--------------注意------------
        print("Loading DD_graph_indicator.txt")
        # 每个节点属于哪个图
        graph_indicator = np.genfromtxt(os.path.join(data_dir, "DD_graph_indicator.txt"),
                                        dtype=np.int64) - 1
        print("Loading DD_graph_labels.txt")
        # 每个图的标签 （2分类 0，1）
        graph_labels = np.genfromtxt(os.path.join(data_dir, "DD_graph_labels.txt"),    #--------------注意------------
                                     dtype=np.int64) - 1
        num_nodes = len(node_labels)  # 节点数 （包含所有图的节点）
        # 通过邻接表生成邻接矩阵  （包含所有的图）稀疏存储节省内存（coo格式 只存储非0值的行索引、列索引和非0值）
        # coo格式无法进行稀疏矩阵运算
        sparse_adjacency = sp.coo_matrix((np.ones(len(adjacency_list)),
                                          (adjacency_list[:, 0], adjacency_list[:, 1])),
                                         shape=(num_nodes, num_nodes), dtype=np.float32)
        print("Number of nodes: ", num_nodes)
        print("Node_labels: ", node_labels)
        print("Graph_labels: ", graph_labels)
        return sparse_adjacency, node_labels, graph_indicator, graph_labels

    def maybe_download(self):
        save_path = os.path.join(self.data_root)
        # 本地不存在 则下载
        if not os.path.exists(save_path):
            self.download_data(self.url, save_path)
        # 对数据集压缩包进行解压
        if not os.path.exists(os.path.join(self.data_root, "DD")):
            zipfilename = os.path.join(self.data_root, "DD.zip")
            with ZipFile(zipfilename, "r") as zipobj:
                zipobj.extractall(os.path.join(self.data_root))
                print("Extracting data from {}".format(zipfilename))

    @staticmethod
    def download_data(url, save_path):
        """数据下载工具，当原始数据不存在时将会进行下载"""
        print("Downloading data from {}".format(url))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 下载数据集压缩包 保存在本地
        data = urllib.request.urlopen(url)
        filename = "DD.zip"
        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(data.read())
        return True


#---图卷积定义---

#输入A,X输出分数

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):

        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        # 权重矩阵
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 使用自定义参数初始化方式

    def reset_parameters(self):  # 自定义权重和偏置的初始化方式
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法"""
        # adjacency (N,N) 归一化的拉普拉斯矩阵
        # input_feature（N,input_dim） N为所有节点个数 （包含所有图）
        support = torch.mm(input_feature, self.weight)  # XW
        output = torch.sparse.mm(adjacency, support)  # L(XW)
        if self.use_bias:
            output += self.bias
        return output  # (N,output_dim=hidden_dim)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


#---数据预处理---

def top_rank(attention_score, graph_indicator, keep_ratio):                         #输入注意力分数z,图片id,池化率k，输出mask
    graph_id_list = list(set(graph_indicator.cpu().numpy()))                        #把每张图的id放进一个列表里
    mask = attention_score.new_empty((0,), dtype=torch.bool)                        #为mask建立空张量，用于保存所有图片的mask表
    for graph_id in graph_id_list:                                                  #遍历每张图
        graph_attn_score = attention_score[graph_indicator == graph_id]             #取出对应id的图的注意力分数
        graph_node_num = len(graph_attn_score)                                      #图的节点数
        graph_mask = attention_score.new_zeros((graph_node_num,), dtype=torch.bool) #为该图所有节点创建一个值全为False的mask张量
        keep_graph_node_num = int(keep_ratio * graph_node_num)                      #计算需要保留的节点数
        _, sorted_index = graph_attn_score.sort(descending=True)                    #对分数降序排序,并得到其索引
        graph_mask[sorted_index[:keep_graph_node_num]] = True                       #把需要保留的节点的mask设置为True
        mask = torch.cat((mask, graph_mask))                                        #把这一批中所有图的掩码拼接到一起
    return mask

def tensor_from_numpy(x, device): #numpy数组转换为tensor 并转移到所用设备上
    return torch.from_numpy(x).to(device)

def normalization(adjacency):                                             #输入邻接矩阵A，输出归一化拉普拉斯矩阵的稀疏格式（非0值的值，非0值的位置，矩阵大小）
    adjacency += sp.eye(adjacency.shape[0])                               #为邻接矩阵添加自连接
    degree = np.array(adjacency.sum(1))                                   #计算度向量
    d_hat = sp.diags(np.power(degree, -0.5).flatten())                    #计算度矩阵
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()                           #计算归一化，有自连接的拉普拉斯矩阵，并转为coo稀疏格式
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()         #得到拉普拉斯矩阵的行索引，列索引
    values = torch.from_numpy(L.data.astype(np.float32))                  #得到拉普拉斯矩阵的非0值
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape) #转换成稀疏矩阵格式
    return tensor_adjacency

def filter_adjacency(adjacency, mask):                     #输入邻接矩阵，mask列表，输出更新后的归一化拉普拉斯矩阵
    device = adjacency.device
    mask = mask.cpu().numpy()
    indices = adjacency.coalesce().indices().cpu().numpy() #邻接矩阵非0值的索引
    num_nodes = adjacency.size(0)                          #节点数
    row, col = indices                                     #邻接矩阵的非0值的行，列索引
    maskout_self_loop = row != col
    row = row[maskout_self_loop]
    col = col[maskout_self_loop]
    sparse_adjacency = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
    filtered_adjacency = sparse_adjacency[mask, :][:, mask] #进行切片
    return normalization(filtered_adjacency).to(device)

#---自注意力池化层---

class SelfAttentionPooling(nn.Module):

    def __init__(self, input_dim, keep_ratio, activation=torch.tanh):#输入维度，存留率，激活函数
        super(SelfAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.keep_ratio = keep_ratio
        self.activation = activation
        self.attn_gcn = GraphConvolution(input_dim, 1)
    def forward(self, adjacency, input_feature, graph_indicator):
        #计算注意力分数
        attn_score = self.attn_gcn(adjacency, input_feature).squeeze()
        attn_score = self.activation(attn_score)
        #获得节点掩码向量
        mask = top_rank(attn_score, graph_indicator, self.keep_ratio)
        #更新邻接矩阵和特征矩阵
        hidden = input_feature[mask] * attn_score[mask].view(-1, 1)
        mask_graph_indicator = graph_indicator[mask]
        mask_adjacency = filter_adjacency(adjacency, mask)
        return hidden, mask_graph_indicator, mask_adjacency

#---读出层---

def global_max_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_max(x, graph_indicator, dim=0, dim_size=num)[0]
def global_avg_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_mean(x, graph_indicator, dim=0, dim_size=num)

# ---SAGPool网络搭建---

class ModelA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):

        super(ModelA, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)

        self.pool = SelfAttentionPooling(hidden_dim * 3, 0.5) #因为拼接了3个卷积层，所以*3

        self.fc1 = nn.Linear(hidden_dim * 3 * 2, hidden_dim) #因为拼接了两个全局池化，所以*2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)



    def forward(self, adjacency, input_feature, graph_indicator):

        gcn1 = F.relu(self.gcn1(adjacency+adjacency*adjacency, input_feature))
        gcn2 = F.relu(self.gcn2(adjacency+adjacency*adjacency, gcn1))
        gcn3 = F.relu(self.gcn3(adjacency+adjacency*adjacency, gcn2))

        gcn_feature = torch.cat((gcn1, gcn2, gcn3), dim=1)  #拼接

        pool, pool_graph_indicator, pool_adjacency = self.pool(adjacency, gcn_feature,graph_indicator) #池化，并获得更新后的特征矩阵，节点索引，邻接矩阵

        readout = torch.cat((global_avg_pool(pool, pool_graph_indicator),global_max_pool(pool, pool_graph_indicator)), dim=1)#读出，将平均值池化和最大值池化拼起来

        fc1 = F.relu(self.fc1(readout))
        fc2 = F.relu(self.fc2(fc1))
        logits = self.fc3(fc2)
        return logits

#
# class ModelB(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_classes=2):
#         """图分类模型结构
#
#         Args:
#         -----
#             input_dim: int, 输入特征的维度
#             hidden_dim: int, 隐藏层单元数
#             num_classes: int, 分类类别数 (default: 2)
#         """
#         super(ModelB, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_classes = num_classes
#
#         # 第一个gcn层 (N,input_dim) ->(N,hidden_dim) N为所有节点数
#         self.gcn1 = GraphConvolution(input_dim, hidden_dim)
#         # 第一个池化层
#         self.pool1 = SelfAttentionPooling(hidden_dim, 0.5)
#         self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
#         self.pool2 = SelfAttentionPooling(hidden_dim, 0.5)
#         self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
#         self.pool3 = SelfAttentionPooling(hidden_dim, 0.5)
#
#         # 把最后的几个全连接层和激活函数 封装在一起
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, num_classes))
#
#     def forward(self, adjacency, input_feature, graph_indicator):
#         # adjacency 拉普拉斯矩阵 （N,N）N为所有节点数
#         # input_feature 所有节点的特征矩阵 (N,input_dim)
#         # (N,input_dim) -> (N,hidden_dim)
#         gcn1 = F.relu(self.gcn1(adjacency, input_feature))
#
#         # gcn1 (N,hidden_dim) adjacency(N,N)  graph_indicator(N,)每个节点属于哪个图
#         # pool1 (N',hidden_dim) N'保留的节点数
#         # pool1_graph_indicator (N',)保留的节点属于哪个图
#         # pool1_adjacency 保留节点的邻接矩阵（归一化）
#         pool1, pool1_graph_indicator, pool1_adjacency = \
#             self.pool1(adjacency, gcn1, graph_indicator)
#
#         # global_pool1 (G,hidden_dim*2)   G为图数
#         global_pool1 = torch.cat(
#             [global_avg_pool(pool1, pool1_graph_indicator),
#              global_max_pool(pool1, pool1_graph_indicator)],
#             dim=1)
#
#         # pool1_adjacency (N',N') N'保留的节点数 新的图结构对应的拉普拉斯矩阵
#         # pool1 (N',hiddem_dim)
#         # (N',hiddem_dim) -> (N',hiddem_dim)
#         gcn2 = F.relu(self.gcn2(pool1_adjacency, pool1))
#
#         # gcn2 (N',hiddem_dim)  pool1_adjacency(N',N')  pool1_graph_indicator(N',)保留的每个节点属于哪个图
#         # pool2 (N'',hidden_dim) N''保留的节点数
#         # pool2_graph_indicator (N'',)保留的节点属于哪个图
#         # pool2_adjacency 保留节点的邻接矩阵（归一化）
#         pool2, pool2_graph_indicator, pool2_adjacency = \
#             self.pool2(pool1_adjacency, gcn2, pool1_graph_indicator)
#
#         # global_pool2 (G,hidden_dim*2)   G为图数
#         global_pool2 = torch.cat(
#             [global_avg_pool(pool2, pool2_graph_indicator),
#              global_max_pool(pool2, pool2_graph_indicator)],
#             dim=1)
#
#         # pool2_adjacency (N'',N'') N''保留的节点数 新的图结构对应的拉普拉斯矩阵
#         # pool2 (N'',hiddem_dim)
#         # (N'',hiddem_dim) -> (N'',hiddem_dim)
#         gcn3 = F.relu(self.gcn3(pool2_adjacency, pool2))
#
#         # gcn3 (N'',hiddem_dim)  pool2_adjacency(N'',N'')  pool2_graph_indicator(N'',)保留的每个节点属于哪个图
#         # pool3 (N''',hidden_dim) N'''保留的节点数
#         # pool3_graph_indicator (N''',)保留的节点属于哪个图
#         # pool3_adjacency 保留节点的邻接矩阵（归一化）
#         pool3, pool3_graph_indicator, pool3_adjacency = \
#             self.pool3(pool2_adjacency, gcn3, pool2_graph_indicator)
#
#         # global_pool3 (G,hidden_dim*2)   G为图数
#         global_pool3 = torch.cat(
#             [global_avg_pool(pool3, pool3_graph_indicator),
#              global_max_pool(pool3, pool3_graph_indicator)],
#             dim=1)
#
#         # readout (G,hidden_dim*2)
#         readout = global_pool1 + global_pool2 + global_pool3
#
#         # logits (G,num_classes=2)
#         logits = self.mlp(readout)
#         return logits

#----------训练预测部分----------

dataset = DDDataset()

# 模型输入数据准备
DEVICE ="cpu"
#所有图对应的大邻接矩阵
adjacency = dataset.sparse_adjacency
#归一化、引入自连接的拉普拉斯矩阵
normalize_adjacency = normalization(adjacency).to(DEVICE)
#所有节点的特征标签
node_labels = tensor_from_numpy(dataset.node_labels, DEVICE)
#把特征标签转换为one-hot特征向量
node_features = F.one_hot(node_labels, node_labels.max().item() + 1).float()
#每个节点对应哪个图
graph_indicator = tensor_from_numpy(dataset.graph_indicator, DEVICE)
#每个图的类别标签
graph_labels = tensor_from_numpy(dataset.graph_labels, DEVICE)
#训练集对应的图索引
train_index = tensor_from_numpy(dataset.train_index, DEVICE)
#测试集对应的图索引
test_index = tensor_from_numpy(dataset.test_index, DEVICE)
#训练集和测试集中的图对应的类别标签
train_label = tensor_from_numpy(dataset.train_label, DEVICE)
test_label = tensor_from_numpy(dataset.test_label, DEVICE)

# 超参数设置
INPUT_DIM = node_features.size(1)
NUM_CLASSES = 2
EPOCHS = 300
HIDDEN_DIM = 32
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001

# 模型初始化
model_g = ModelA(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
model = model_g
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss().to(DEVICE)
# Adam优化器
optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#设置空列表记录精度
train_acc_all=[]
loss_all=[]
x=[]

#训练模型

model.train()
for epoch in range(EPOCHS):
    logits = model(normalize_adjacency, node_features, graph_indicator)  # 对所有数据(图)前向传播 得到输出
    loss = criterion(logits[train_index], train_label)  # 只对训练的数据计算损失值
    optimizer.zero_grad()
    loss.backward()  # 反向传播计算参数的梯度
    optimizer.step()  # 使用优化方法进行梯度更新
    # 训练集准确率
    train_acc = torch.eq(
        logits[train_index].max(1)[1], train_label).float().mean()
    print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}".format(epoch, loss.item(), train_acc.item()))
    train_acc_all.append(train_acc.item())
    loss_all.append(loss.item())

# for i in range(EPOCHS):
#     x.append(float(i))
#
# d={'epoch1':x,'acc1':train_acc_all}
# path='D:/mat/path1.mat'
# scio.savemat(path,d)

plt.figure()
plt.plot(train_acc_all,"r-")
plt.title("train acc ")
plt.show()

#测试模型

model.eval()  # 测试模式
with torch.no_grad():  # 关闭求导
    logits = model(normalize_adjacency, node_features, graph_indicator)  # 所有数据前向传播
    test_logits = logits[test_index]  # 取出测试数据对应的输出
    # 计算测试数据准确率
    test_acc = torch.eq(
        test_logits.max(1)[1], test_label
    ).float().mean()

print("TestAcc",test_acc.item())
