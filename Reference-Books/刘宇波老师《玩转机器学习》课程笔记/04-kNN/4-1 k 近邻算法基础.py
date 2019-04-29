import numpy as np
import matplotlib.pyplot as plt

raw_data_X = [[3.39353321, 2.33127338],
              [3.11007348, 1.78153964],
              [1.34380883, 3.36836095],
              [3.58229404, 4.67917911],
              [2.28036244, 2.86699026],
              [7.42343694, 4.69652288],
              [5.745052, 3.5339898],
              [9.17216862, 2.51110105],
              [7.79278348, 3.424088894],
              [7.93982082, 0.79163723]]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

# 描点绘图

# 待预测点
X = np.array([8.093607318, 3.365731514])

plt.rcParams['font.sans-serif'] = ['STHeiti']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='r', label='类别 0')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='b', label='类别 1')
plt.scatter(X[0], X[1], color='g', label='待预测的数据点')
plt.legend(loc='best')
plt.show()

distances = [np.linalg.norm(point - X) for point in X_train]
print("打印每个点距离待测点的距离：")
for index, distance in enumerate(distances):
    print("[{}] {}".format(index, np.round(distance, 2)))

sorted_index = np.argsort(distances)
print(y_train[sorted_index])

k = 6
topK = y_train[sorted_index][:k]
print(topK)

from collections import Counter

votes = Counter(topK)
mc = votes.most_common(n=1)
print(mc)
print("根据投票得出的点 X 的标签为：", mc[0][0])
