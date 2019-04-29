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


def mean_squared_error(y_test, y_predict):
    return np.sum((y_test - y_predict) ** 2) / len(y_test)


def root_mean_squared_error(y_test, y_predict):
    return np.sqrt(np.sum((y_test - y_predict) ** 2) / len(y_test))


def mean_absolute_error(y_test, y_predict):
    return np.sum(np.abs(y_test - y_predict)) / len(y_test)


def r2_score(y_test, y_predict):
    return 1 - mean_squared_error(y_test, y_predict) / np.var(y_test)


def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))


def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))


def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))


def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))


# 混淆矩阵
def confusion_matrix(y_true, y_predict):
    return np.array([
        [TN(y_true, y_predict), FP(y_true, y_predict)],
        [FN(y_true, y_predict), TP(y_true, y_predict)]
    ])


# 精准率
def precision_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.


# 召回率
def recall_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.


# f1_score
def f1_score(y_true, y_predict):
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    try:
        return 2 * precision * recall / (precision + recall)
    except:
        return 0


# TPR
def TPR(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.


# FPR
def FPR(y_true, y_predict):
    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)
    try:
        return fp / (fp + tn)
    except:
        return 0.
