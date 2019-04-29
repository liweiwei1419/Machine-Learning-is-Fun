from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# 超参数以列表形式给出，列表里面的单个元素是一个 dict
param_grid = [
    {
        'weights': ['uniform'],
        'n_neighbors': [i for i in range(1, 11)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)],
        'p': [i for i in range(1, 6)]

    }
]

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=666)

knn_clf = KNeighborsClassifier()
# n_jobs = -1 将计算机所有的核都用于并行计算
# verbose 在搜索的过程中进行一些输出，整数值越大，输出的信息越详细，通常选择 2
grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)
# estimator 评估量
# 网格搜索最佳分类器的超参数的值
print(grid_search.best_estimator_)

knn_clf = grid_search.best_estimator_
knn_clf.predict(X_test)
knn_clf.score(X_test, y_test)
