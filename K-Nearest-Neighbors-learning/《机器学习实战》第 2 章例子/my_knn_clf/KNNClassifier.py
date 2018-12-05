import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler


# 使用 kNN 算法要计算距离，为了避免距离被大的量纲单位主导，要做归一化 auto_norm
# 归一化数据：把特征都缩放到统一尺度可以比较，为了不让计算距离的结果被特征的数字差值最大的属性所主导

class KNNClassifier:

    def __init__(self, k):
        assert type(k) == int and k >= 1, "k 必须是整数，并且大于等于 1"
        self.k = k
        # 不希望被外界调用，所以使用单下划线前缀
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, predict_point):
        distances = [np.linalg.norm(predict_point - x_train) for x_train in self._X_train]
        # 距离从近到远排序
        nearest = np.argsort(distances)
        # 找最近的 k 个
        top_k_y = [self._y_train[i] for i in nearest[:self.k]]
        # 进行计数
        votes = Counter(top_k_y)
        return votes.most_common(1)[0][0]


if __name__ == '__main__':
    def create_data_set():
        X = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        y = ['A', 'A', 'B', 'B']
        return X, y


    def auto_norm(X):
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        scaler.fit(X)
        return scaler.transform(X)


    X, y = create_data_set()
    X_scaler = auto_norm(X)

    knn_clf = KNNClassifier(k=3)
    knn_clf.fit(X, y)
    result = knn_clf.predict([[0, 0]])
    print(result)
