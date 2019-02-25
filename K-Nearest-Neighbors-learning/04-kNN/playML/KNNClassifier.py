import numpy as np
from collections import Counter
from playML.mymetrics import accuracy_score


# 要想让编译器编译通过，要将 04-kNN 标记为源代码文件夹
# 这个模块模仿了 scikit-learn 的编码风格，编写了 k 近邻算法的简单实现

class KNNClassifier:
    def __init__(self, k):
        """
        初始化 kNN 分类器
        :param k: 超参数 k
        """
        assert k >= 1, "k must be valid"
        self.k = k
        # 不希望被外界调用，所以使用单下划线前缀
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """
        根据训练数据集 X_train 和 y_train 训练 kNN 分类器，
        可以看到这个 fit 没有显示的训练过程，只是无脑地把训练数据集放进私有变量
        :param X_train:
        :param y_train:
        :return:
        """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k."
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """
        给定待预测数据集 X_predict ，返回表示 X_predict 的结果向量
        :param X_predict: 待预测的数据集，可以是列表形式
        :return:
        """
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """
        给单个的待预测数据进行预测"
        :param x:
        :return:
        """
        assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of x must be equal to X_train"
        distances = [np.linalg.norm(x - x_train) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def score(self, x_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(x_test)
        return accuracy_score(y_test, y_predict)

    # 如果你在 Python 解释器里直接敲 a 后回车，调用的是 a.__repr__() 方法
    def __repr__(self):
        return "KNN(k=%d)" % self.k
