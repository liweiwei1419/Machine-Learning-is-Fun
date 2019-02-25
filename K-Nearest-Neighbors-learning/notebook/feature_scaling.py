# 做 k 近邻分类的时候，不要忘记数据标准化

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=666)

standardScaler = StandardScaler()

# 这里不用 y_train
# X_train 本身是没有改变的
standardScaler.fit(X_train)
print(standardScaler.mean_)
print(standardScaler.scale_)

X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train_standard, y_train)
print(knn_clf.predict(X_test_standard))
print(y_test)

score = knn_clf.score(X_test_standard, y_test)
print(score)