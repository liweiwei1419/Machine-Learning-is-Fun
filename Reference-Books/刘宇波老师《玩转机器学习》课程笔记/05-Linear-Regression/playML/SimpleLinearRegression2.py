import numpy as np


class SimpleLinearRegression2:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, '简单线性回归只能处理一个特征的训练数据集'
        assert len(x_train) == len(y_train), 'the size of x_train must be equal to the size of y_train'
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        self.a_ = (x_train - x_mean).dot(y_train - y_mean)/(x_train - x_mean).dot(x_train - x_mean)
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """给定单个待预测数据x，返回x的预测结果值"""
        return self.a_ * x_single + self.b_

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return self.r2_score(y_test, y_predict)

    def mean_square_error(self, y_true, y_predict):
        assert len(y_true) == len(y_predict), "the size of y_true must be equal to the size of y_predict"
        return (y_true - y_predict).dot(y_true - y_predict) / len(y_true)

    def r2_score(self, y_true, y_predict):
        return 1 - self.mean_square_error(y_true, y_predict) / np.var(y_true)

    def __repr__(self):
        return 'SimpleLinearRegression()'
