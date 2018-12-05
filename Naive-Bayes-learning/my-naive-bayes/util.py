import re
import random


# 默认设置训练集占整个数据集的比例 0.7;


class FileOperate:

    def __init__(self, data_path, label):
        self.data_path = data_path
        self.label = label

    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as fr:
            content = fr.readlines()
            print("一共 {} 条数据。".format(len(content)))

        X = list()
        y = list()

        for line in content:
            result = line.split(self.label, maxsplit=2)
            X.append(FileOperate.__clean_data(result[1]))
            y.append(1 if result[0] == 'spam' else 0)

        return X, y

    @staticmethod
    def __clean_data(origin_info):
        """
        清洗数据，去掉非字母的字符，和字节长度小于 2 的单词
        :return:
        """
        # 先转换成小写
        # 把标点符号都替换成空格
        temp_info = re.sub('\W', ' ', origin_info.lower())
        # 根据空格（大于等于 1 个空格）
        words = re.split(r'\s+', temp_info)
        return list(filter(lambda x: len(x) >= 3, words))


def train_test_split(X, y, test_size=0.3, random_state=42):
    # 设置随机数种子
    random.seed(random_state)

    all_len = len(X)
    indexes = list(range(0, all_len))
    random.shuffle(indexes)
    test_len = int(all_len * test_size)

    test_indexes = indexes[:test_len]
    train_indexes = indexes[test_len:]

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for index in test_indexes:
        X_test.append(X[index])
        y_test.append(y[index])
    for index in train_indexes:
        X_train.append(X[index])
        y_train.append(y[index])

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    data_path = '../input/SMSSpamCollection'
    label = '\t'
    fo = FileOperate(data_path, label)
    X, y = fo.load_data()
    print(X)
