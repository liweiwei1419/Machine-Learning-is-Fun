import numpy as np


def train_test_split(X, y, test_size=0.3, random_state=42):
    # 设置随机数种子
    np.random.seed(random_state)
    all_len = X.shape[0]
    indexes = np.arange(all_len)
    np.random.shuffle(indexes)
    test_len = int(all_len * test_size)
    X_train = X.iloc[test_len:, :]
    X_test = X.iloc[:test_len, :]
    y_train = y[test_len:]
    y_test = y[:test_len]
    return X_train, X_test, y_train, y_test
