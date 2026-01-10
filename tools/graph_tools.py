import torch
from torch_geometric.data import Data
def build_graph_corr(train_dataset):
    """
    构建属性之间基于皮尔逊相关系数的加权图
    节点为属性，边为属性之间的皮尔逊相关系数，权重 ∈ [-1, 1]
    """
    dataset_size = len(train_dataset)
    attribute_num = len(train_dataset.attrs)

    # Y: [N, A]，每行是一个样本的属性多热向量
    attr_tensor_list = []
    for sample in train_dataset.data:

        image, attrs, obj,attr_list,pair_list = sample  # attr_label 是一个 multi-hot tensor
        attr_label = torch.Tensor(attr_list)
        attr_tensor_list.append(attr_label.unsqueeze(0))

    Y = torch.cat(attr_tensor_list, dim=0)  # [N, A]
    n, m = Y.shape

    # 均值中心化
    Y_mean = Y.float().mean(dim=0, keepdim=True)
    Y_centered = Y.float() - Y_mean  # [N, A]

    # 协方差矩阵 [A, A]
    cov_matrix = (Y_centered.T @ Y_centered) / n  # [A, A]

    # 方差
    var = torch.diag(cov_matrix).unsqueeze(0)  # [1, A]
    std = var.sqrt()

    # 避免除以0
    corr_matrix = cov_matrix / (std.T @ std + 1e-9)  # [A, A]

    # 构造图的边 (i, j) 和对应权重
    edge_index = []
    edge_weight = []

    threshold = 0.1
    for i in range(attribute_num):
        for j in range(attribute_num):
            if i != j and abs(corr_matrix[i, j]) > threshold:
                edge_index.append([i, j])
                edge_weight.append(corr_matrix[i, j])

    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_weight = torch.tensor(edge_weight)

    graph = Data(edge_index=edge_index, edge_weight=edge_weight)

    return edge_index, edge_weight, graph


def build_graph_condition_probability(train_dataset):
    """
    构建属性之间基于皮尔逊相关系数的加权图
    节点为属性，边为属性之间的皮尔逊相关系数，权重 ∈ [-1, 1]
    """
    dataset_size = len(train_dataset)
    attribute_num = len(train_dataset.attrs)

    # Y: [N, A]，每行是一个样本的属性多热向量
    attr_tensor_list = []
    for sample in train_dataset.data:

        image, attrs, obj,attr_list,pair_list = sample  # attr_label 是一个 multi-hot tensor
        attr_label = torch.Tensor(attr_list)
        attr_tensor_list.append(attr_label.unsqueeze(0))

    Y = torch.cat(attr_tensor_list, dim=0)  # [N, A]
    n, m = Y.shape

  
    # C: 计算属性共现矩阵 [A, A]
    C = (Y.T @ Y)  # 每个 c_mn 是属性 m 和 n 同时为1的次数

    cmm = torch.diag(C).unsqueeze(1)

    # 条件概率矩阵 A: a_mn = P(n | m) = c_mn / c_mm
    A = C / cmm  # shape: [A, A]

    edge_index = A.nonzero(as_tuple=False).t()          # shape [2, E]
    edge_weight = A[edge_index[0], edge_index[1]]       # shape [E]

    return edge_index,A ,Data(edge_index=edge_index, edge_weight=edge_weight)
