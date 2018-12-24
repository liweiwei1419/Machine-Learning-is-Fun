# 朴素贝叶斯

# 创建数据集
def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 这些文本的类别由人工标注，我们的代码就要"学习"这些数据
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is abusive（辱骂的）, 0 not
    return posting_list, class_vec


def create_vocab_list(dataset):
    vocab_set = set([])
    for document in dataset:
        # 只有两个 Set 之间才可以进行 | （取并集）操作
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


# 词集模型
# 注意：只能传一个句子进去，得到一个句子向量
# 一个句子中如果有重复的单词，只记录一次，因为上面已经有 set 的作用
# 一个句子中如果有重复的单词，只记录一次，因为上面已经有 set 的作用
# 一个句子中如果有重复的单词，只记录一次，因为上面已经有 set 的作用
def set_of_word2Vec(vocab_list, input_set):
    '''
    返回一个向量，在词集模型模型中，一个单词至多出现一次
    :param vocab_list: 所有句子组成的单词集合
    :param input_set: 一个句子的所有单词构成的集合
    :return: 一个句子的词向量
    '''
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("单词 %s 没有出现在单词表中" % word)
    return return_vec


import numpy as np


def trainNB0(train_matrix, train_category):
    '''

    :param train_matrix:
    :param train_category:
    :return:

    p_abusive 含有侮辱性词汇的句子的概率
    '''
    # 训练集有多少条数据
    num_train_docs = len(train_matrix)
    # 单词表中有多少个单词，即每个词向量的维度
    num_words = len(train_matrix[0])
    # 含有 abusive 性质的文档的概率
    # 不含有 abusive 性质的文档的概率用 1 去减就可以得到了
    p_abusive = np.sum(train_category) / float(num_train_docs)
    # p0_num = np.zeros(num_words)  # 不含有 abusive 性质
    p0_num = np.ones(num_words)  # 不含有 abusive 性质
    # p1_num = np.zeros(num_words)  # 含有 abusive 性质
    p1_num = np.ones(num_words)  # 含有 abusive 性质

    # p0_denom = 0.0 # 作为分母的项
    # p1_denom = 0.0 # 作为分母的项
    p0_denom = 2.0  # 作为分母的项
    p1_denom = 2.0  # 作为分母的项

    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]  # 向量
            p1_denom += sum(train_matrix[i])  # 数，累加上每一个句子（词向量）包含的单词个数
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    # 在含有侮辱性质的文档中，每个单词出现的概率
    p1_vec = np.log(p1_num / p1_denom)  # 以后我们会看到，这个结果会被对数函数作用
    p0_vec = np.log(p0_num / p0_denom)  # 以后我们会看到，这个结果会被对数函数作用
    return p1_vec, p0_vec, p_abusive


# 这里就是使用朴素贝叶斯那个最最基本的公式进行比较，谁大就判别属于哪个类
def classifyNB(ver2classify, p0_ver, p1_ver, p_class1):
    p1 = np.sum(ver2classify * p1_ver) + np.log(p_class1)
    p_class0 = 1 - p_class1
    p0 = np.sum(ver2classify * p0_ver) + np.log(p_class0)
    if p1 > p0:
        return 1
    else:
        return 0
