# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/02
    项目名称： K-means 模型的手工实现
    参考：     https://gist.github.com/iandanforth/5862470
"""

import math
import random


class Cluster(object):
    """
        聚类
    """

    def __init__(self, samples):
        if len(samples) == 0:
            # 如果聚类中无样本点
            raise Exception("错误：一个空的聚类！")

        # 属于该聚类的样本点
        self.samples = samples

        # 该聚类中样本点的维度
        self.n_dim = samples[0].n_dim

        # 判断该聚类中所有样本点的维度是否相同
        for sample in samples:
            if sample.n_dim != self.n_dim:
                raise Exception("错误： 聚类中样本点的维度不一致！")

        # 设置初始化的聚类中心
        self.centroid = self.cal_centroid()

    def __repr__(self):
        """
            输出对象信息
        """
        return str(self.samples)

    def update(self, samples):
        """
            计算之前的聚类中心和更新后聚类中心的距离
        """

        old_centroid = self.centroid
        self.samples = samples
        self.centroid = self.cal_centroid()
        shift = get_distance(old_centroid, self.centroid)
        return shift

    def cal_centroid(self):
        """
           对于一组样本点计算其中心点
        """
        n_samples = len(self.samples)
        # 获取所有样本点的坐标（特征）
        coords = [sample.coords for sample in self.samples]
        unzipped = zip(*coords)
        # 计算每个维度的均值
        centroid_coords = [math.fsum(d_list)/n_samples for d_list in unzipped]

        return Sample(centroid_coords)


class Sample(object):
    """
        样本点类
    """
    def __init__(self, coords):
        self.coords = coords    # 样本点包含的坐标
        self.n_dim = len(coords)    # 样本点维度

    def __repr__(self):
        """
            输出对象信息
        """
        return str(self.coords)


def get_distance(a, b):
    """
        返回样本点a, b的欧式距离
        参考：https://en.wikipedia.org/wiki/Euclidean_distance#n_dimensions
    """
    if a.n_dim != b.n_dim:
        # 如果样本点维度不同
        raise Exception("错误: 样本点维度不同，无法计算距离！")

    acc_diff = 0.0
    for i in range(a.n_dim):
        square_diff = pow((a.coords[i]-b.coords[i]), 2)
        acc_diff += square_diff
    distance = math.sqrt(acc_diff)

    return distance


def gen_random_sample(n_dim, lower, upper):
    """
        生成随机样本
    """
    sample = Sample([random.uniform(lower, upper) for _ in range(n_dim)])
    return sample
