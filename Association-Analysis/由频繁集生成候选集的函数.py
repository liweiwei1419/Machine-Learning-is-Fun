def apriori_gen(frequent_set):
    """
    修改以后的由频繁集生成候选集函数 liwei 2018-10-26
    :param frequent_set:
    :return:
    """
    # 频繁集中的每个项集所包含的元素个数
    m = len(frequent_set[0])
    # 频繁集的元素个数，主要用于得到下标
    l = len(frequent_set)
    candidate_set_list = []
    for i in range(l - 1):
        for j in range(i + 1, l):
            # 取交集
            intersection = frequent_set[i] & frequent_set[j]
            # 例如：{3,5} 和 {2,5}，它们的交集是 {5} ，才有必要合并
            if len(intersection) == (m - 1):
                # 取并集
                union_set = frequent_set[i] | frequent_set[j]
                if union_set not in candidate_set_list:
                    candidate_set_list.append(union_set)
    # 去重以后作为列表返回
    return candidate_set_list


L2 = [{5, 3}, {5, 10}]
print(apriori_gen(L2))

L0 = [{0}, {1}, {2}, {3}]
print(apriori_gen(L0))
print('-' * 20)

L1 = [{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}]
print(apriori_gen(L1))

L2 = [{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}]
print(apriori_gen(L2))
