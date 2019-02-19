from collections import defaultdict

# 参考资料：《机器学习实战》第 10 章
# 《使用 Apriori 算法和 FP-growth 算法进行关联分析》
# https://www.cnblogs.com/qwertWZ/p/4510857.html

class Apriori:

    def __init__(self, min_support=0.5, min_confidence=0.7):
        """
        :param min_support: 最小支持度
        :param min_confidence: 最小置信度
        """
        self.min_support = min_support
        self.min_confidence = min_confidence

        # 临时变量
        self.data_set = None
        self.data_set_len = 0

        # 频繁项集列表，是一个层层递进的关系
        # 第 1 层，频繁项集的元素个数均为 1
        # 第 2 层，频繁项集的元素个数均为 2
        self.fre_item_sets_list_ = []

        # 关联规则的集合
        self.association_rules_ = []
        # 各个项集的支持度
        self.supports_ = dict()

    def fit(self, data_set):
        self.data_set = data_set
        self.data_set_len = float(len(self.data_set))

        # 第 1 部分：从事务列表中找到频繁项集
        # 首先生成项集个数为 1 的候选集
        C1 = self.__create_C1()
        # print('C1', C1)
        # 根据项集个数为 1 的候选集生成频繁项集列表
        self.__apriori(C1)

        # 第 2 部分：根据频繁项集和支持度字典得到关联规则
        # 从频繁项集中挖掘关联规则，项集元素为 1 的就不考虑了

        self.__generate_association_rules()

    def __generate_association_rules(self):
        frequent_set_len = len(self.fre_item_sets_list_)
        # print(frequent_set_len)
        for i in range(1, frequent_set_len):
            for fre_item_sets in self.fre_item_sets_list_[i]:
                self.__generate_association_rules_from_one_frequent_set(fre_item_sets)
        # print(fre_item_sets)
        # 根据每一个元素个数大于 2 的频繁项集，挖掘关联规则

    def __generate_association_rules_from_one_frequent_set(self, frequent_set):
        """
        从 1 个频繁项集（项集个数必须大于 1）挖掘关联关系
        :param frequent_set:
        :return:
        """
        # print("开始对频繁项集 {} 挖掘关联规则。".format(frequent_set))

        # 这里的 H 可以理解为右边列表
        H = [frozenset([item]) for item in frequent_set]
        # 这里的 H 可以理解为右边列表
        # print("H", H)
        # 右边列表的每个元素的数量
        m = len(H[0])

        # 只要这个频繁集的元素个数比右边列表中的元素个数多，就可以尝试挖掘关联关系

        # 下面这一行的逻辑有些绕，但是结合挖掘关联关系的那张表，就不难理解

        while len(frequent_set) > m:
            # 说明可以挖掘关联关系，但不一定可以挖掘出关联关系
            # 根据右边列表和频繁集列表计算置信度
            H = self.__calc_confidence(frequent_set, H)
            # print('H', H)
            # 得到新的右边列表，生成下一级候选集

            # 右边列表只有 1 个的时候，不能生成下一级候选集
            # 因此判断条件就是 len(H)>1
            if len(H) > 1:
                # 这里的 H 可以理解为候选的右边列表
                H = self.__apriori_gen(H)
                m += 1
            else:
                break

    def __calc_confidence(self, fre_item_sets, H):
        """
        根据右边列表和频繁集列表计算置信度
        :param fre_item_sets:
        :param H: 右边列表，右边列表可以认为是待推荐的项集列表
        :return:
        """

        # 下一层关联规则的计算要用到的频繁项集列表，同样它出现在右边
        recommended_items = []
        for item_sets in H:
            # 对于右边列表的每一项，计算置信度
            # 把置信度的定义弄清楚以后，这一行代码就很清晰了
            # 置信度其实是个条件概率，
            # fre_item_sets - item_sets 是前件
            # item_set 是后件
            confidence = self.supports_[fre_item_sets] / self.supports_[fre_item_sets - item_sets]

            # print('置信度：', confidence)
            if confidence >= self.min_confidence:
                recommended_items.append(item_sets)
                # 记录关联关系
                # 是一个 tuple，第 1 个元素是前件，第 2 个元素是后件，第 3 个元素是置信度
                self.association_rules_.append((fre_item_sets - item_sets, item_sets, confidence))
        return recommended_items

    def __apriori(self, C1):
        """
        由数据集生成频繁项集，保存于 self.frequent_set_list_，
        生成频繁项集的过程中保存各个候选项集的支持度，保存于 self.supports_ = dict()。
        :param data_set:
        :return:
        """
        # 首先由数据集生成第 1 层候选集
        Ck = C1
        while True:
            # 由候选集生成频繁集
            # 这一步要遍历整个数据集，并且是在一个循环中进行的，因此是个耗时的操作
            # FP-growth 算法优化了找频繁项集的过程
            Lk = self.__scan_data_set(Ck)
            # 候选集要过滤才能得到频繁集，所以频繁集有可能为空
            # 如果过滤剩下的频繁集为空，就可以退出循环了

            if len(Lk) == 0:
                break

            # 在这一层找到的频繁项集应该添加到结果集中
            self.fre_item_sets_list_.append(Lk)

            # 由频繁集生成候选集，只要频繁集非空，候选集一定非空
            Ck = self.__apriori_gen(Lk)

    def __create_C1(self):
        # candidate 候选集
        # c1 表示这个候选集中的所有项集所包含的元素个数为 1 个，
        # 例如候选项集：[[1], [3], [4], [2], [5]]
        C1 = []
        for transaction in self.data_set:
            for item in transaction:
                if [item] not in C1:
                    C1.append([item])
        # 在计算支持度的时候，要把项集放入一个 dict，此时就要进行 hash 操作
        # 作为 key ，得是不可变对象，还需要满足是集合，因此要使用 frozenset 对象
        # TypeError: unhashable type: 'list'

        return list(map(frozenset, C1))

    def __scan_data_set(self, Ck):
        """
        将候选项集中的每一个项集，去遍历整个事务集，计算每个项集的支持度计数
        根据最小支持度的设置，去掉小于最小支持度的项集，剩下来的就是频繁项集
        :param Ck:
        :return:
        """
        # 使用原生的 dict，当 key 不存在的时候，设置对应的 value 为 1
        # 当 key 存在的时候，设置对应的 value 为（原来的 value + 1）
        # 使用 defaultdict 就不用做判断了
        item_sets_count = defaultdict(int)

        for fre_item_sets in Ck:
            for transaction in self.data_set:
                # issubset 是 frozenset 的一个方法，用于判断一个集合是不是另一个集合的子集
                # 用在这里是恰到好处的
                if fre_item_sets.issubset(transaction):
                    item_sets_count[fre_item_sets] += 1

        # print(item_sets_count)

        # 当前的频繁集列表
        current_frequent_set_list = []
        # 根据最小支持度设置，保留符合要求的项集，即频繁集

        for item_sets, count in item_sets_count.items():
            # 计算项集的支持度
            support = count / self.data_set_len
            # 计算好每个项集的支持度以后，就保存到支持度字典中，以后计算置信度的时候再来查询
            self.supports_[item_sets] = support
            if support >= self.min_support:
                current_frequent_set_list.append(item_sets)
        return current_frequent_set_list

    def __apriori_gen(self, frequent_set_list):
        """
        由频繁集列表生成候选集，这部分与《机器学习实战》这本书上的算法不同。
        :param frequent_set_list:
        :return:
        """
        # 频繁集中的每个项集所包含的元素个数
        m = len(frequent_set_list[0])
        # 频繁集的元素个数，主要用于得到下标
        l = len(frequent_set_list)
        candidate_set_list = []
        for i in range(l - 1):
            for j in range(i + 1, l):
                # 取交集
                intersection = frequent_set_list[i] & frequent_set_list[j]
                # 例如：{3,5} 和 {2,5}，它们的交集是 {5} ，才有必要合并
                if len(intersection) == (m - 1):
                    # 取并集
                    union_set = frequent_set_list[i] | frequent_set_list[j]
                    # 去重
                    if union_set not in candidate_set_list:
                        candidate_set_list.append(union_set)

        return candidate_set_list


if __name__ == '__main__':
    # 模拟生成示例数据
    data_set = [
        [1, 3, 4],
        [2, 3, 5],
        [1, 2, 3, 5],
        [2, 5]
    ]
    apriori = Apriori()
    apriori.fit(data_set)

    print('得到的频繁集列表如下：')
    for frequent_set in apriori.fre_item_sets_list_:
        print(frequent_set)
    #
    # print('得到的项集支持度字典如下：')
    # for item_set, support in apriori.supports_.items():
    #     print(item_set, '支持度：', support)

    print('挖掘出的关联规则如下：')
    for rule in apriori.association_rules_:
        print(rule)
