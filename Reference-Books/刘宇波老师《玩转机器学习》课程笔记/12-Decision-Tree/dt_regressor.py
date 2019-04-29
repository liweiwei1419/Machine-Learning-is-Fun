import numpy as np

from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def create_data(n):
    '''
    产生用于回归问题的数据集
    :param n:  数据集容量
    :return: 返回一个元组，元素依次为：训练样本集、测试样本集、训练样本集对应的值、测试样本集对应的值
    '''
    np.random.seed(0)
    X = 5 * np.random.rand(n, 1)
    y = np.sin(X).ravel()
    noise_num = int(n / 5)
    y[::5] += 3 * (0.5 - np.random.rand(noise_num))  # 每第5个样本，就在该样本的值上添加噪音
    # 拆分原始数据集为训练集和测试集，其中测试集大小为元素数据集大小的 1/4
    return model_selection.train_test_split(X, y, test_size=0.25,
                                            random_state=1)


def test_DecisionTreeRegressor(*data):
    X_train, X_test, y_train, y_test = data
    regr = DecisionTreeRegressor()
    regr.fit(X_train, y_train)
    print('训练数据集评分 {}'.format(regr.score(X_train, y_train)))
    print('测试数据集评分 {}'.format(regr.score(X_test, y_test)))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    # [:,np.newaxis] 使得原来一行的数据变成了一列
    X = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y = regr.predict(X)
    # 描点训练数据
    ax.scatter(X_train, y_train, label='train sample', c='g')
    # 描点测试数据
    ax.scatter(X_test, y_test, label='test sample', c='r')
    ax.plot(X, y, label='predict_value', linewidth=2, alpha=0.5)
    ax.set_label('data')
    ax.set_label('target')
    ax.set_title('Decision Tree Regression')
    ax.legend()
    plt.show()


def test_DecisionTreeRegressor_splitter(*data):
    X_train, X_test, y_train, y_test = data
    # 指定切分原则：best 表示选择最优的切分
    # random 表示随机切分
    splitters = ['best', 'random']

    for splitter in splitters:
        regr = DecisionTreeRegressor(splitter=splitter)
        regr.fit(X_train, y_train)
        print('splitter:{}'.format(splitter))
        print('训练数据集评分 {}'.format(regr.score(X_train, y_train)))
        print('测试数据集评分 {}'.format(regr.score(X_test, y_test)))


def test_DecisionTreeRegressor_depth(*data, max_depth):
    X_train, X_test, y_train, y_test = data
    depths = np.arange(1, max_depth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        regr = DecisionTreeRegressor(max_depth=depth)
        regr.fit(X_train, y_train)
        training_scores.append(regr.score(X_train, y_train))
        testing_scores.append(regr.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(depths,training_scores,label='training scores')
    ax.plot(depths,testing_scores,label='testing scores')
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha=0.5)
    plt.show()
