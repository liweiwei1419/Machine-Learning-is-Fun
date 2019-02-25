import numpy as np


def accuracy_score(y_true, y_predict):
    '''
    计算 y_true 和 y_predict 之间的准确率
    :param y_true:
    :param y_predict:
    :return:
    '''
    assert len(y_true) == len(y_predict), "the size of y_true must be equal to the size of y_predict"
    return sum(y_true == y_predict) / len(y_true)

def mean_squared_error(y_true, y_predict):
    """计算 y_true 和 y_predict 之间的 MSE"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"
    return np.sum((y_true - y_predict)**2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """计算 y_true 和 y_predict 之间的 RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """计算 y_true 和 y_predict 之间的 RMSE"""
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)

def r2_score(y_true, y_predict):
    """计算 y_true 和 y_predict 之间的 R Square"""
    return 1 - mean_squared_error(y_true, y_predict)/np.var(y_true)
