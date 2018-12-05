import numpy as np
from collections import Counter


def kNN_classify(k, X_train, y_train, x):
    """
    :param k: 超参数 k
    :param X_train: 训练数据集的特征（是一个矩阵）
    :param y_train: 训练数据集的标签（是一个向量）
    :param x: 待预测的数据
    :return: 列表的标签
    """
    # k 的范围不能超过训练样本的个数
    assert 1 <= k <= X_train.shape[0], 'k must be valid'
    # 训练数据集 特征矩阵的样本数 应该和 训练数据集 标签向量的维度一样
    assert X_train.shape[0] == y_train.shape[0], 'the size of X_train must equal to the size of y_train'
    # 训练数据集 特征矩阵的特征数 应该和 预测数据的维度一样
    assert X_train.shape[1] == x.shape[0], 'the feature number of x must be equal to X_train'

    distances = [np.linalg.norm(x_train - x) for x_train in X_train]
    # 按照从小到大排好的参数列表
    nearest = np.argsort(distances)
    # 获得一个训练数据集的标签列表，然后进行投票
    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)
    return votes.most_common(1)[0][0]
