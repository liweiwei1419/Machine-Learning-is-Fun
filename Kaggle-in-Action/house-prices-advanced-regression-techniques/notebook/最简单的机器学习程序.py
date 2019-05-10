from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
iris = load_iris()

X = iris.data
y = iris.target

# 把 random_state 调整到 42 的时候，在测试数据集上的表现可以达到 100%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=666)

# 这是一个分类算法，而且是机器学习领域非常著名的分类算法——逻辑回归，也叫对数几率回归
lr = LogisticRegression()
lr.fit(X_train, y_train)
score1 = lr.score(X_test, y_test)
print("模型在测试数据集上的得分：", score1)

y_pred = lr.predict(X_test)
score2 = accuracy_score(y_test, y_pred)
print("准确率：", score2)
