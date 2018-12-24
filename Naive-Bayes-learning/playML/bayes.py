import math


class NaiveBayes:
    """
    根据贝叶斯定理，我们只要分别统计出先验概率和各自类别下每个单词的条件概率即可，
    实际上就是进行 word count 。
    """

    def __init__(self):
        self.len = 0
        self.__ham_count = 0  # 非垃圾短信数量
        self.__spam_count = 0  # 垃圾短信数量

        # 注意：这里一定要选列表，不能使用 set，因为是计算词频，肯定是需要有重复的，才能计算概率分布
        self.__ham_words = list()  # 非垃圾短信单词列表
        self.__spam_words = list()  # 垃圾短信单词列表

        # 训练集中不重复单词集合
        self.__word_dictionary_set = set()
        self.__word_dictionary_size = 0

        self.__ham_map = dict()  # 非垃圾短信的词频统计
        self.__spam_map = dict()  # 垃圾短信的词频统计

        self.__ham_probability = 0  # 非垃圾短信的先验概率
        self.__spam_probability = 0  # 垃圾短信的先验概率

    def fit(self, X_train, y_train):
        """
        单词字典完全由训练数据集构建
        :param X_train:
        :param y_train:
        :return:
        """
        self.len = len(X_train)
        self.build_word_set(X_train, y_train)
        self.word_count()

    def predict(self, X_train):
        return [self.predict_one(sentence) for sentence in X_train]

    def build_word_set(self, X_train, y_train):
        """
        第 1 步：建立单词字典
        :param X_train:
        :param y_train:
        :return:
        """
        for words, y in zip(X_train, y_train):
            if y == 0:
                # 非垃圾短信
                # self.__ham_count += 1
                for word in words:
                    self.__ham_words.append(word)
                    self.__word_dictionary_set.add(word)
            if y == 1:
                # 垃圾短信
                self.__spam_count += 1
                for word in words:
                    self.__spam_words.append(word)
                    self.__word_dictionary_set.add(word)

        # print('非垃圾短信数量', self.__ham_count)
        # print('垃圾短信数量', self.__spam_count)
        # print(self.__word_dictionary_set)
        self.__ham_count = self.len - self.__spam_count
        self.__word_dictionary_size = len(self.__word_dictionary_set)

    def word_count(self):
        # 第 2 步：不同类别下的词频统计
        for word in self.__ham_words:
            self.__ham_map[word] = self.__ham_map.setdefault(word, 0) + 1

        for word in self.__spam_words:
            self.__spam_map[word] = self.__spam_map.setdefault(word, 0) + 1

        # 下面两行计算先验概率

        # 垃圾短信的先验概率
        self.__spam_probability = self.__spam_count / self.len

        # 非垃圾短信的先验概率
        self.__ham_probability = 1 - self.__spam_probability
        # self.__spam_probability = self.__spam_count / (self.__ham_count + self.__spam_count)

    def predict_one(self, sentence):
        ham_pro = 0
        spam_pro = 0

        for word in sentence:
            # print('word', word)
            ham_pro += math.log(
                (self.__ham_map.get(word, 0) + 1) / (self.__ham_count + self.__word_dictionary_size))

            spam_pro += math.log(
                (self.__spam_map.get(word, 0) + 1) / (self.__spam_count + self.__word_dictionary_size))

        ham_pro += math.log(self.__ham_probability)
        spam_pro += math.log(self.__spam_probability)

        # print('垃圾短信概率', spam_pro)
        # print('非垃圾短信概率', ham_pro)
        return int(spam_pro >= ham_pro)
