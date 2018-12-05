from bayes import NaiveBayes
from util import FileOperate
from util import train_test_split
from metrics import accuracy_score

# 运行这部分代码的时候，要将 my-naive-bayes 这个文件夹设置为源代码的根文件夹

if __name__ == '__main__':
    # 1、加载数据，spam 表示垃圾短信（1），ham 表示非垃圾短信（0）
    data_path = '../input/SMSSpamCollection'
    label = '\t'
    fo = FileOperate(data_path, label)
    X, y = fo.load_data()

    # 2、分割数据集，得到训练数据集与测试数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=666)

    # 开始训练
    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    # 开始预测
    y_pred = nb.predict(X_test)

    # 计算得分
    score = accuracy_score(y_test, y_pred)
    print('准确率：', score)
