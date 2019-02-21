from collections import defaultdict



class Apriori:
    def __init__(self, min_support=0.5, min_conf=0.7):
        self.min_support = min_support
        self.min_conf = min_conf

        # 所有在计算过程中出现的项集的支持度
        self.item_sets_support_rate = dict()

    def fit(self, data_set):
        # 3、生成频繁项，这里设置最小支持度为 0.5
        # frequent_item_sets 所有频繁项集
        # support_data
        frequent_item_sets, support_data = self.apriori(data_set)
        # print('所有的频繁项集：')
        # for item in frequent_item_sets:
        #     print(item)

        # 寻找强关联规则，最小置信度设置为 0.7
        rules = self.generate_rules_right(frequent_item_sets, support_data)
        # print('找到的强关联规则：',rules)
        return rules

    # 生成候选项
    # c1 是元素的个数为 1 的所有候选项的集合
    def create_c1(self, data_set):
        c1 = []
        for transaction in data_set:
            for item in transaction:
                if [item] not in c1:
                    c1.append([item])
        c1.sort()
        # c1 是一个 list，把这个 list 中的每个元素都包装成为一个 frozenset({2})
        return list(map(frozenset, c1))

    # 生成频繁项
    # 在生成频繁项集的过程中，可以计算出支持度
    # Ck 是候选项集，是一个集合构成的 list，这个集合中的元素个数都一样，是 k
    def scan_d(self, D, Ck):
        '''
        :param D: 真个数据集
        :param Ck: k 表示项集的个数
        :param min_support: 最小支持度
        :return:频繁项集
        '''
        ss_cnt = defaultdict(int)
        # 这一步，对于候选项集中的每一个项集，都要遍历整个数据集 D，效率低，
        # 可以使用 FP-growth 算法发现频繁项集
        for tid in D:
            for can in Ck:
                # print('can--', type(can), 'tid--', tid)
                # issubset 是 frozenset 的语法
                if can.issubset(tid):
                    # print('是子集')
                    ss_cnt[can] += 1
        # 总的事务数
        num_items = float(len(D))

        # 频繁项集列表
        frequent_item_sets = []
        # dict：{项集:支持度}
        support_data = {}
        for key, value in ss_cnt.items():
            support = value / num_items
            support_data[key] = support
            if support >= self.min_support:
                # 只有是频繁项集，才添加到这个列表
                frequent_item_sets.insert(0, key)
        return frequent_item_sets, support_data

    # 由频繁项集生成候选项集
    def apriori_gen(self, Lk, k):
        """
        该函数通过频繁项集列表 Lk 和项集个数 k 生成候选项集 Ck+1 。

        合并的前提是：在排序过后，前 k-2 项都相同，只有最后一项不同的时候，才合并，合并以后的项集个数，就是 (k-2)+2 = k。

        :param Lk:频繁项集列表。
        :param k: 生成的项集集合中，每个项集的元素个数。即单个项集的个数。
        :return:
        """
        # 生成的候选集放在这里
        ret_list = []
        # Lk 是低一层的频繁集列表
        len_lk = len(Lk)
        # print("LK", Lk)
        for i in range(len_lk):
            for j in range(i + 1, len_lk):
                # 前 k-2 项相同时，将两个集合合并
                L1 = list(Lk[i])[:k - 2]
                L2 = list(Lk[j])[:k - 2]
                L1.sort()
                L2.sort()
                # 只要是前缀一样，就合并
                if L1 == L2:
                    ret_list.append(Lk[i] | Lk[j])
        # print('ret_list', ret_list, end='\n\n')
        return ret_list

    # apriori 是整个算法的核心
    # 生成频繁项集（分层）和项集的支持度，用于计算关联规则
    def apriori(self, data_set):
        """

        :param data_set: 数据集
        :return: L：包含整个数据集的频繁项集的列表，support_data 支持度字典（除了频繁项集，还有非频繁项集）
        """
        # 2、生成候选项
        # 生成只有 1 个元素的所有项（不是项集，因为只有 1 个元素）的列表
        C1 = self.create_c1(data_set)
        # print('C1', C1)
        # [frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]
        # data_set 里面每一项都是一个 list，把它转成 set
        D = list(map(set, data_set))
        L1, support_data = self.scan_d(D, C1)
        L = [L1]
        k = 2
        while len(L[k - 2]) > 0:  # 只要当前一层有元素，就可以生成下一层，按照这种逻辑，【最后一层一定是空的】
            Ck = self.apriori_gen(L[k - 2], k)  # 这里 k 表示当前生成的项集中的元素个数
            # scan_d 这个方法返回的 Lk 只是频繁项集，supK 不只是频繁项集
            # 对于候选集中的每一个项集，都要去遍历整个数据集，
            Lk, supK = self.scan_d(D, Ck)

            # 如果 support_data 中有和 supK 的 key 相同的 ，用 supK 的 value 去覆盖它。即以 supK 计算出来的为准
            support_data.update(supK)  # 更新一下 项集-支持度 键值对

            L.append(Lk)
            k += 1
        return L, support_data

    # 生成关联规则的主函数
    def generate_rules(self, L, support_data):
        """

        :param L: 【频繁项集】
        :param support_data:支持度键值对
        :param min_conf:
        :return:
        """

        # 找到的强关联规则列表
        big_rule_list = []
        # 这个集合在递归的过程中，是一直要用上的

        # 频繁项集的第 1 层是 [frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})] ，只有 1 个元素
        # 这个不用看了，因为我们不关心只有一个元素的项集，因此不从 0 开始
        # print(L[0])
        for i in range(1, len(L)):
            # 针对频繁项集某一层中的每个频繁项集
            # print("L[i]", L[i])
            for freq_set in L[i]:
                # 频繁项集中的单个元素组成的列表
                # print(freq_set)
                H1 = [frozenset([item]) for item in freq_set]

                # 这个 H1 集合中的项集只有一个元素，并且只会出现在关联规则的右边

                if i > 1:
                    # print('H1',H1)
                    # 当项集的个数为 3 的时候，走这个分支，所以写在前面
                    # 根据当前候选规则集 H 生成下一层候选规则集
                    self.rules_from_conseq(freq_set, H1, support_data, big_rule_list)

                else:
                    # print('H1',H1)

                    # 当项集的个数只有 2 的时候，走这一支，其余情况走上一个分支
                    # 只计算置信度
                    self.calc_confidence(freq_set, H1, support_data, big_rule_list)
        return big_rule_list

    # 生成关联规则的主函数(正确的逻辑)
    def generate_rules_right(self, L, support_data):
        big_rule_list = []
        for i in range(1, len(L)):
            for freq_set in L[i]:
                H1 = [frozenset([item]) for item in freq_set]
                print('H1\t\t', H1, 'freq_set', freq_set)
                self.rules_from_conseq_right(freq_set, H1, support_data, big_rule_list)
        return big_rule_list

    def rules_from_conseq_right(self, freq_set, H, support_data, brl):

        m = len(H[0])
        # print('H[0]\t', H[0], end='\n\n')

        # print('m', m, freq_set)
        if len(freq_set) > m:
            print('走到这里')
            hmp1 = self.calc_confidence(freq_set, H, support_data, brl)
            if len(hmp1) > 1:
                # 递归调用
                hmp1 = self.apriori_gen(hmp1, m + 1)
                self.rules_from_conseq_right(freq_set, hmp1, support_data, brl)

    # 计算规则的置信度
    def calc_confidence(self, freq_set, H, support_data, brl):
        """
        对候选规则进行评估
        :param freq_set: 对单个频繁项集进行评估，
        :param H:
        :param support_data:
        :param brl:
        :return:
        """
        # 未修剪的
        pruned_h = []

        # H 中的项集的元素个数都只有 1 个
        # print('H',H)
        # print()

        # conseq 表示可以出现在关联规则右侧的元素列表
        for conseq in H:
            # 置信度的计算
            # print('--freq_set', freq_set, 'conseq', conseq, )
            # 关联规则是这样的： {freq_set - conseq } -> {conseq}

            conf = support_data[freq_set] / support_data[freq_set - conseq]
            # print(freq_set - conseq, '-->', conseq, 'conf:', conf)
            if conf >= self.min_conf:
                # 符合最小置信度要求的，记录
                # print(freqSet - conseq, '-->', conseq, 'conf:', conf)
                brl.append((freq_set - conseq, conseq, conf))
                # 在满足最小置信度的前提下
                # 即如果顾客选择了 freq_set - conseq ，那么推荐 conseq

                # 将可以出现在右侧的项集保存
                pruned_h.append(conseq)

        return pruned_h

    # 根据当前候选规则集 H 生成下一层候选规则集
    def rules_from_conseq(self, freq_set, H, support_data, brl):
        # print('H', H, 'freq_set', freq_set)
        # m 表示这一层频繁集的元素的个数，因为都一样，看第 1 个就可以了

        # 函数先计算H中的频繁项集大小 m。
        # 接下来查看该频繁项集是否大到可以移除大小为 m 的子集。
        # 如果可以的话，则将其移除。
        # 使用函数 aprioriGen() 来生成 H 中元素的无重复组合，结果保存在 Hmp1中，这也是下一次迭代的 H 列表。

        m = len(H[0])
        if len(freq_set) > (m + 1):
            hmp1 = self.apriori_gen(H, m + 1)
            hmp1 = self.calc_confidence(freq_set, hmp1, support_data, brl)
            if len(hmp1) > 1:
                # 递归调用
                self.rules_from_conseq(freq_set, hmp1, support_data, brl)


# 模拟生成示例数据
def load_data_set():
    return [
        [1, 3, 4],
        [2, 3, 5],
        [1, 2, 3, 5],
        [2, 5]
    ]


if __name__ == '__main__':
    # 模拟生成数据
    data_set = load_data_set()
    apriori = Apriori(min_support=0.5, min_conf=0.7)
    rules = apriori.fit(data_set)

    print("找到的关联规则如下：")
    for rule in rules:
        print(rule)

    # (frozenset({1}), frozenset({3}), 1.0)
    # {1,3}/{1}
    # 表示在已有的历史购买记录中，购买 1 的用户，100% 购买了 3
    # 这意味着，可以给欲购买 1 的用户，推荐 3

    # frozenset({3}) --> frozenset({1}) conf: 0.6666666666666666
    # 这意味着 {1,3}/{3} ，即在购买了 3 的记录中，66.7% 购买了 1
    # 因此，设置了最小置信度为 0.7 的情况下，欲购买 3 的用户，不用推荐 1
