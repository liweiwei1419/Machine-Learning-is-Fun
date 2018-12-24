# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/02
    项目名称： K-means 模型的手工实现
    参考：     https://gist.github.com/iandanforth/5862470
"""
import random
from kmeans_tools import Cluster, get_distance, gen_random_sample
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def kmeans(samples, k, cutoff):
    """
        kmeans函数
    """

    # 随机选k个样本点作为初始聚类中心
    init_samples = random.sample(samples, k)

    # 创建k个聚类，聚类的中心分别为随机初始的样本点
    clusters = [Cluster([sample]) for sample in init_samples]

    # 迭代循环直到聚类划分稳定
    n_loop = 0
    while True:
        # 初始化一组空列表用于存储每个聚类内的样本点
        lists = [[] for _ in clusters]

        # 开始迭代
        n_loop += 1
        # 遍历样本集中的每个样本
        for sample in samples:
            # 计算样本点sample和第一个聚类中心的距离
            smallest_distance = get_distance(sample, clusters[0].centroid)
            # 初始化属于聚类 0
            cluster_index = 0

            # 计算和其他聚类中心的距离
            for i in range(k - 1):
                # 计算样本点sample和聚类中心的距离
                distance = get_distance(sample, clusters[i+1].centroid)
                # 如果存在更小的距离，更新距离
                if distance < smallest_distance:
                    smallest_distance = distance
                    cluster_index = i + 1

            # 找到最近的聚类中心，更新所属聚类
            lists[cluster_index].append(sample)

        # 初始化最大移动距离
        biggest_shift = 0.0

        # 计算本次迭代中，聚类中心移动的距离
        for i in range(k):
            shift = clusters[i].update(lists[i])
            # 记录最大移动距离
            biggest_shift = max(biggest_shift, shift)

        # 如果聚类中心移动的距离小于收敛阈值，即：聚类稳定
        if biggest_shift < cutoff:
            print("第{}次迭代后，聚类稳定。".format(n_loop))
            break
    # 返回聚类结果
    return clusters


def run_main():
    """
        主函数
    """
    # 样本个数
    n_samples = 1000

    # 特征个数 (特征维度)
    n_feat = 2

    # 特征数值范围
    lower = 0
    upper = 200

    # 聚类个数
    n_cluster = 3

    # 生成随机样本
    samples = [gen_random_sample(n_feat, lower, upper) for _ in range(n_samples)]

    # 收敛阈值
    cutoff = 0.2

    clusters = kmeans(samples, n_cluster, cutoff)

    # 输出结果
    for i, c in enumerate(clusters):
        for sample in c.samples:
            print('聚类--{}，样本点--{}'.format(i, sample))

    # 可视化结果
    plt.subplot()
    color_names = list(mcolors.cnames)
    for i, c in enumerate(clusters):
        x = []
        y = []
        random.choice
        color = [color_names[i]] * len(c.samples)
        for sample in c.samples:
            x.append(sample.coords[0])
            y.append(sample.coords[1])
        plt.scatter(x, y, c=color)
    plt.show()

if __name__ == '__main__':
    run_main()
