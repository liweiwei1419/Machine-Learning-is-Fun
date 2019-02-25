# 手写实现准确度的计算


def accuracy_score(y_true, y_predict):
    """
    计算 y_true 和 y_predict 之间的准确率
    :param y_true:
    :param y_predict:
    :return:
    """
    assert len(y_true) == len(y_predict), "the size of y_true must be equal to the size of y_predict"
    return sum(y_true == y_predict) / len(y_true)
