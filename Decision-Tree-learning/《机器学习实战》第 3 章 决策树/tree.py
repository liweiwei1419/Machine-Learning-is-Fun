from collections import defaultdict
from math import log

import pandas as pd
import matplotlib.pyplot as plt

from plot_view import create_plot


class DecisionTreeClassifier:

    def __init__(self):
        plt.rcParams['font.sans-serif'] = ['STHeiti']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        self.data_set = None
        self.feature_names = None
        self.target_name = None
        self.tree_dict = None

    def fit(self, X_train, y_train):
        self.data_set = pd.concat([X_train, y_train], axis=1)
        self.feature_names = X_train.columns.values.tolist()
        self.target_name = y_train.name
        self.tree_dict = DecisionTreeClassifier.create_tree(self.data_set, self.feature_names[:], self.target_name)

    @staticmethod
    def calc_shannon_entropy(data_set, column_name):
        """
        取出一列计算香农熵
        :param data_set:
        :param column_name:
        :return:
        """
        y = data_set[column_name].values
        d = defaultdict(int)
        for item in y:
            d[item] += 1
        m = len(y)
        shannon_ent = 0
        for value in d.values():
            pro = value / m
            shannon_ent += (-pro * log(pro, 2))
        return shannon_ent

    @staticmethod
    def calc_conditional_entropy(data_set, column_name, target_name):
        """
        计算条件熵
        """
        l = data_set.shape[0]
        conditional_entropy = 0.0
        for name, values in data_set.groupby(column_name):
            pro = len(values) / l
            shannon_entropy = DecisionTreeClassifier.calc_shannon_entropy(values, target_name)
            conditional_entropy += (pro * shannon_entropy)
        return conditional_entropy

    @staticmethod
    def choose_best_feature_to_split(data_set, target_name):
        """
        根据【信息增益比】选择最优的特征
        :param data_set:
        :return:
        """
        # print('此时特征个数:', data_set.shape[1] - 1)

        info_gain_ratio_max = 0
        # 划分方式的特征索引
        best_feature = None

        shannon_entropy = DecisionTreeClassifier.calc_shannon_entropy(data_set, target_name)

        for col in data_set.columns[:-1]:
            cond_entropy = DecisionTreeClassifier.calc_conditional_entropy(data_set=data_set, column_name=col,
                                                                           target_name=target_name)
            # 计算特征熵
            feature_entropy = DecisionTreeClassifier.calc_shannon_entropy(data_set, col)
            info_gain_ratio = (shannon_entropy - cond_entropy) / feature_entropy
            # print('{:10} 信息增益比：{:.4f}'.format(col, info_gain_ratio))

            if info_gain_ratio_max < info_gain_ratio:
                info_gain_ratio_max = info_gain_ratio
                best_feature = col

        return best_feature

    @staticmethod
    def create_tree(data_set, feature_names, target_name):
        # 当前节点中包含的样本完全属于同一类别，无需划分（递归返回出口1）
        if len(data_set[target_name].value_counts()) == 1:
            # 只有 1 个类了，就返回第 1 个数据的 target
            # print(data_set['classes'])
            return data_set[target_name].iloc[0]

        # 当前特征集合为空，无法划分（递归返回出口2）
        # 【返回样本数最多的类别】
        if data_set.shape[1] == 1:
            # value_counts 这个方法默认设置了 sort=True
            # 样本数最多的类别排在了第 1 个
            return data_set[target_name].value_counts(sort=True)[0]

        # 当前节点包含的样本集合为空，不能划分，此时不会生成新的节点（递归返回出口3）
        if data_set.shape[0] == 0:
            return

        best_feature = DecisionTreeClassifier.choose_best_feature_to_split(data_set=data_set, target_name=target_name)
        # print('选择的最佳特征：', best_feature)
        decision_tree = {best_feature: {}}
        # 当前最佳特征选定以后，就从特征列表中删除

        feature_names.remove(best_feature)

        # 下面根据当前的最佳特征划分数据集
        # 【这是一个递归创建的过程】

        for value, group in data_set.groupby(best_feature):
            del group[best_feature]
            sub_column_names = feature_names[:]
            decision_tree[best_feature][value] = DecisionTreeClassifier.create_tree(group, sub_column_names,
                                                                                    target_name)
        return decision_tree

    def plot(self):
        """
        绘图
        :return:
        """
        create_plot(self.tree_dict)

    def pre_recursion(self, in_tree, test_vec):
        first_str = list(in_tree.keys())[0]
        second_dict = in_tree[first_str]
        fea_index = self.feature_names.index(first_str)

        for key in second_dict.keys():
            if test_vec[fea_index] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    class_label = self.pre_recursion(second_dict[key], test_vec)
                else:
                    class_label = second_dict[key]
        return class_label

    def predict_one(self, test_vec):
        return self.pre_recursion(self.tree_dict, test_vec)

    def predict(self, X_test):
        return [self.predict_one(item) for item in X_test.values]


if __name__ == '__main__':
    data = pd.read_csv('./loan_example.csv')
    target_name = data.columns[-1]
    feature_names = data.columns[:-1].values.tolist()

    dt = DecisionTreeClassifier()

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    dt.fit(X, y)

    # dt.plot()

    y_pred = dt.predict(X)

    print(y_pred)
    print(y.values)
