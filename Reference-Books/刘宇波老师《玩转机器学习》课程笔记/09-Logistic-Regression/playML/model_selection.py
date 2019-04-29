# 将原有的训练数据集中的一部分用作测试数据集
import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):
    '''
    将数据 X 和 y 按照 test_radio 分割成
    X_train,y_train,X_test,y_test
    :param X:
    :param y:
    :param test_radio:
    :param seed:
    :return:
    '''

    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ration must be valid"

    if seed:
        np.random.seed(seed)
    # 得到 0 到 len(X) - 1 的一个排列，相当于打乱了数据
    shuffled_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)

    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]
    # 特别注意返回的数据集的顺序
    return X_train, X_test, y_train, y_test
