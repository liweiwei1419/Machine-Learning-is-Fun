# 生成候选集 c1，即项集元素为 1 的所有项集的集合
def create_c1(data_set):
    c1 = []
    for transaction in data_set:
        for item in transaction:
            if [item] not in c1:
                c1.append([item])
    c1.sort()
    # c1 是一个 list，把这个 list 中的每个元素都包装成为一个 frozenset({2})
    return list(map(frozenset, c1))


data_set = [
    [1, 3, 4],
    [2, 3, 5],
    [1, 2, 3, 5],
    [2, 5]]

result = create_c1(data_set)
print(result)
