# 使用信息熵寻找最优的划分
# 本例子中，特征是连续的，与我们一开始学习决策树的例子不同
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from math import log

from sklearn import datasets

iris = datasets.load_iris()
# 只选择了两个特征
# 索引为 2 和 3 的特征
X = iris.data[:, 2:]
y = iris.target

# criterion 评判标准
dt_clf = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=100)
dt_clf.fit(X, y)


# 绘制决策边界的函数
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


plot_decision_boundary(dt_clf, axis=[0.5, 7.5, 0, 3])
plt.title("决策边界是横平竖直的")
plt.rcParams['font.sans-serif'] = ['STHeiti']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.scatter(X[y == 2, 0], X[y == 2, 1])
plt.show()


# 实现使用信息熵进行划分

# 假设我们已经选定了
def split(X, y, d, value):
    '''
    d：第几个 feature
    value：value 作为划分的 value，小于等于这个 value 归为一类，大于这个 value 归为一类
    '''
    index_a = (X[:, d] <= value)
    index_b = (X[:, d] > value)
    return X[index_a], X[index_b], y[index_a], y[index_b]


def entropy(y):
    counter = Counter(y)
    res = 0.0
    for num in counter.values():
        # 计算这个类别的概率
        p = num / len(y)
        res += -p * log(p)
    return res


# 我们这里只创建二叉树
def try_split(X, y):
    '''
    每一轮寻找最优的特征和特征划分的值
    :param X:
    :param y:
    :return:
    '''
    best_entropy = float('inf')
    # 最佳的 feature
    best_d = -1
    # 最佳的划分值
    best_v = -1
    for d in range(X.shape[1]):
        sorted_index = np.argsort(X[:, d])
        for i in range(1, len(X)):
            if X[sorted_index[i], d] != X[sorted_index[i - 1], d]:
                v = (X[sorted_index[i], d] + X[sorted_index[i - 1], d]) / 2
                X_l, X_r, y_l, y_r = split(X, y, d, v)
                # 注意理解这一行代码：
                # 我们回忆一下，信息熵的定义，
                # 信息熵是数据不确定性的度量，信息熵越大，数据的不确定性越大，信息熵越小，数据的确定性越大。
                # 划分好以后的数据集的信息熵一定要是最小的，才能保证在这个特征的这个划分下，不确定性越小。
                e = entropy(y_l) + entropy(y_r)
                if e < best_entropy:
                    best_entropy = e
                    best_d = d
                    best_v = v
    return best_entropy, best_d, best_v


best_entropy, best_d, best_v = try_split(X, y)
print("最小的信息熵", best_entropy)
print("最小的信息熵的特征对应第几个索引", best_d)
print("最小的信息熵的特征的划分阈值", best_v)
