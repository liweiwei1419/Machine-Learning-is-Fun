from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=666)

print(X_train.min(axis=0))
print(X_train.max(axis=0))

clf = KNeighborsClassifier()

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# clf.fit(X_train_scaler, y_train)
# score = clf.score(X_test_scaler, y_test)
# print(score)

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)
