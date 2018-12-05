from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=666)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train_scaler, y_train)

X_test_scaler = scaler.transform(X_test)
y_pred = clf.predict(X_test_scaler)
score = clf.score(X_test_scaler, y_test)
print(score)

acc = accuracy_score(y_test, y_pred)
print(acc)
