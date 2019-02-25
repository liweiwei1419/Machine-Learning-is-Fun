# 在 scikit-learn 中使用 k 近邻算法
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# k 近邻算法也可以用于分类
# from sklearn.neighbors import KNeighborsRegressor
# 这里的 n_neighbors 是一个超参数

# k 近邻算法严重依赖训练数据集，可以认为训练数据集就是模型本身

# 原始的
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

kNN_classifier = KNeighborsClassifier(n_neighbors=6)
kNN_classifier.fit(X_train, y_train)

X = np.array([8.093607318, 3.365731514])
y_predict = kNN_classifier.predict(X.reshape(1, -1))
print(y_predict[0])
