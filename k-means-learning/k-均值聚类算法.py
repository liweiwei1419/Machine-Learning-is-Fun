import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs

# 计算得分
# 计算分类标签
# 计算聚类中心

X, y = make_blobs(n_samples=200,
                  n_features=2,
                  centers=4,  # 4 个聚类中心
                  cluster_std=1,  # 每个聚类的标准差
                  center_box=(-10.0, 10.0),  # 每个聚类中心坐标的边界值
                  shuffle=True,
                  random_state=666)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print("这是一个负数的得分：")
print("得分:", kmeans.score(X))  # 得分: -563.8089024201704
# scikit-learn 里的计算方法是：所有的点到其聚类中心的距离的总和

print(kmeans.cluster_centers_)
print(kmeans.labels_)

scores = []

for point, label in zip(X, kmeans.labels_):
    # 看源码知道，这里计算距离的时候要平方
    scores.append(np.linalg.norm(point - kmeans.cluster_centers_[label]) ** 2)
print(-np.sum(scores))  # -563.8089024201704

# 可以对聚类的总数 k 进行遍历，得到一个 score 最小的值，这里当然是 k = 4 是最合适的
# 用途：可以对文档进行聚类
