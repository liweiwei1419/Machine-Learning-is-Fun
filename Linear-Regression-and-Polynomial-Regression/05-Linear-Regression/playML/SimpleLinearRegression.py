import numpy as np


class SimpleLinearRegression:

    def __init__(self):
        """初始化Simple Linear Regression模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        '''
        根据训练数据集的特征和标签计算回归系数
        :param x_train:
        :param y_train:
        :return:
        '''

        # Simple Linear Regressor can only solve single feature training data.
        assert x_train.ndim == 1, '简单线性回归只能处理一个特征的训练数据集'
        assert len(x_train) == len(y_train), 'the size of x_train must be equal to the size of y_train'
        # 根据公式，先计算平均值
        # 公式要自己手写推导一下
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # 分子
        num = 0.0
        # 分母
        d = 0.0
        for x_i, y_i in zip(x_train, y_train):
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) * (x_i - x_mean)
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        # 固定的写法
        return self

    def predict(self, x_predict):
        '''
        给定待预测数据集 x_predict，返回表示 x_predict 的结果向量
        :param x_predict: 是一个列表，列表中的每一个数分别代入回归方程中，得到一个结果向量
        :return:
        '''
        # 深刻理解 assert x_predict.ndim == 1 的作用
        assert x_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"
        return np.array([self._predict(x) for x in x_predict])

    # single 单一的
    def _predict(self, x_single):
        """给定单个待预测数据x，返回x的预测结果值"""
        return self.a_ * x_single + self.b_

    # 用测试数据集进行评分
    def score(self, x_test, y_test):
        '''
        根据测试数据集计算当前模型的准确度
        :param x_test:
        :param y_test:
        :return:
        '''

        y_predict = self.predict(x_test)
        return self.r2_score(y_test, y_predict)

    def mean_square_error(self, y_true, y_predict):
        '''
        计算 y_true 和 y_predict 之间均方误差（MSE）
        :param y_true:
        :param y_predict:
        :return:
        '''
        # 还可以有好几种计算方法
        assert len(y_true) == len(y_predict), "the size of y_true must be equal to the size of y_predict"
        return (y_true - y_predict).dot(y_true - y_predict) / len(y_true)

    def r2_score(self, y_true, y_predict):
        '''
        计算 y_true 和 y_predict 之间的R Square
        :param y_true: 测试数据集中的 target，这部分的值是正确的，所以可以命名成 true
        :param y_predict:  我们通过回归方程预测出来的数据，所以用 y_predict
        :return:
        '''
        return 1 - self.mean_square_error(y_true, y_predict) / np.var(y_true)

    def __repr__(self):
        return 'SimpleLinearRegression()'
