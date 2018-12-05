from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

news = fetch_20newsgroups(subset='all')

X = news.data
y = news.target

# 训练数据集和测试数据集分离
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=666)

# 数据预处理
count_vet = CountVectorizer()

X_count_train = count_vet.fit_transform(X_train)
X_count_test = count_vet.transform(X_test)

# 使用多项式分布（独立重复 n 次投骰子）
mnb_count = MultinomialNB()
mnb_count.fit(X_count_train, y_train)
y_count_predict = mnb_count.predict(X_count_test)

print(classification_report(y_test, y_count_predict, target_names=news.target_names))
