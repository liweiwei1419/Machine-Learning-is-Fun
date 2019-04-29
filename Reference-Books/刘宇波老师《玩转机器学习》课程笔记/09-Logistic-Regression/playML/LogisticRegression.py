import numpy as np
from playML.metrics import accuracy_score


# 逻辑回归的代码是由线性回归改造而来
class LogisticRegression:
    def __init__(self):
        self.coef_ = None  # 权重系数
        self.intercept_ = None  # 偏置
        self._theta = None

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    # 使用梯度下降法求得的 theta 值，即最佳参数
    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练 Logistic Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 逻辑回归的目标函数：损失函数的期望（平均值）
        def J(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                # 除以样本的总数，是希望梯度不受样本的数量影响
                return - np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(X_b)
            except:
                return float('inf')

        # 推导出的梯度函数
        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            i_iter = 0

            # 多写几遍就顺溜了
            # 当迭代次数达到最大迭代次数 或者 相邻两次迭代的
            while i_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                # 不能写成 theta -= eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                i_iter += 1
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict_proba(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((X_predict.shape[0], 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_predict, y_test)

    def __repr__(self):
        return 'LogisticRegression()'
