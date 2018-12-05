import numpy as np


class Perceptron:
    """
    感知机分类器：假设数据集是线性可分的
    """

    def __init__(self, eta=0.01, n_iter=10):
        """

        :param eta: 学习率，between 0.0 and 1.0，float
        :param n_iter: 最大迭代次数，int
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        # 同李航《统计学习方法》P29
        # "1" 表示偏置，即如果变量有 2 个，学习的权重就会有 3 个
        # 感知机就是学习这一组参数向量
        # 这里 y 只有两个取值，1 或者 -1
        # target - self.predict(xi)，predict 函数返回 1 或者 -1
        # 如果相同，则上式 = 0，即分类正确的点对权重更新没有帮助
        self.w_ = np.zeros(1 + X.shape[1])

        print(self.w_)
        self.errors_ = []

        for _ in range(self.n_iter):
            print('迭代次数', _)
            # 表示这一轮分错的数据的个数
            errors = 0
            # 把所有的数据都看一遍
            for xi, target in zip(X, y):
                # 【注意】这个处理就包括了 target 和 self.predict 相等的情况，
                # 如果相等，下面两行 self.w_[1:] 和 self.w_[0] 都不会更新
                # 如果不等，相当于朝着父梯度方向走了一点点
                # 随机梯度下降法，每次只使用一个数据更新权重

                # print('实际',target,'预测',self.predict(xi))
                if target == self.predict(xi):
                    continue
                update = self.eta * target
                # w
                self.w_[1:] += update * xi
                # b
                self.w_[0] += update

                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        计算输出
        :param X:
        :return:
        """
        # X 是 m * n 型
        # self.w_[1:] 是 n * 1 型，可以 dot
        # + self.w_[0] 发生了广播
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        预测类别变量，只返回 1 或者 -1
        :param X:
        :return:
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
