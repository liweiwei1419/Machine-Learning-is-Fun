import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import  datasets
from sklearn import model_selection
import matplotlib.pyplot as plt



def load_data():
    irir = datasets.load_iris()
    X = irir.data
    y = irir.target
    return model_selection.train_test_split(X,y,test_size=0.25,random_state=0,stratify=y)

def test_decision_tree_classifier(*data):
    X_train,X_test,y_train,y_test =data
    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    print('训练数据集评分 {}'.format(clf.score(X_train, y_train)))
    print('测试数据集评分 {}'.format(clf.score(X_test, y_test)))


def test_decision_tree_classifier_criterion(*data):
    X_train,X_test,y_train,y_test =data
    criterions = ['entropy','gini']
    for criterion in criterions:
        clf = DecisionTreeClassifier(criterion=criterion)
        clf.fit(X_train,y_train)
        print('criterion {}'.format(criterion))
        print('训练数据集评分 {}'.format(clf.score(X_train, y_train)))
        print('测试数据集评分 {}'.format(clf.score(X_test, y_test)))

def test_decision_tree_classifier_splitter(*data):
    X_train,X_test,y_train,y_test =data
    splitters = ['best','random']
    for splitter in splitters:
        clf = DecisionTreeClassifier(splitter=splitter)
        clf.fit(X_train,y_train)
        print('splitter {}'.format(splitter))
        print('训练数据集评分 {}'.format(clf.score(X_train, y_train)))
        print('测试数据集评分 {}'.format(clf.score(X_test, y_test)))

def test_decision_tree_classifier_depth(*data, max_depth):
    X_train, X_test, y_train, y_test = data
    depths = np.arange(1, max_depth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train, y_train))
        testing_scores.append(clf.score(X_test, y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(depths,training_scores,label='training scores',marker='o')
    ax.plot(depths,testing_scores,label='testing scores',marker='*')
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha=0.5,loc='best')
    plt.show()