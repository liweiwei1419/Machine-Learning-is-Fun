
# 参考资料：使用 Apriori 算法和 FP-growth 算法进行关联分析
# https://www.cnblogs.com/qwertWZ/p/4510857.html

# 代码还是基于《机器学习实战》，只不过没有封装

# FP树的数据结构
class TreeNode:
    def __init__(self, name, count, parent_node):
        # 接下来的 4 个属性是在 FP 树中才用到的
        self.name = name
        self.count = count
        self.parent = parent_node
        self.children = {}

        # 这个
        self.next_node = None

    def inc(self, count):
        """
        给当前结点增加指定的次数
        :param count:
        :return:
        """
        self.count += count

    # 打印自己这棵树
    def display(self, depth=1):
        print('--' * depth, self.name, '', self.count)
        for child in self.children.values():
            child.display(depth + 1)


# 构建FP树
def create_tree(data_set, min_sup=1):
    header_table = {}
    # 这里为了和后面构建条件树的代码兼容，弄得复杂了一些，其实就是在做 word count
    for trans in data_set:
        for item in trans:
            header_table[item] = header_table.get(item, 0) + data_set[trans]
            # 初始化的时候 data_set[trans] 都是 1
            # 在构建条件 FP 树的时候 data_set[trans] 的值才是频数

    # 把不符合最小支持度要求的项去掉，得到头指针表
    header_table = dict(filter(lambda x: x[1] >= min_sup, header_table.items()))

    # 如果头指针表没有元素，则返回空
    # 头指针表的所有键合起来，就是单个元素的频繁项集
    # 我们就从单个元素的频繁项集开始递归查找项集
    freq_item_set = set(header_table.keys())
    if len(freq_item_set) == 0:
        # 返回结点和头指针列表
        return None, None

    for k in header_table:
        header_table[k] = [header_table[k], None]

    # 上面的代码，是给头指针列表增加一个位置
    # r [3, None]
    # z [5, None]
    # y [3, None]
    # x [4, None]
    # t [3, None]
    # s [3, None]

    # FP 树的根结点，是一个空节点
    ret_tree = TreeNode('Null Set', 0, None)

    for tran_set, count in data_set.items():
        # 给每一条事务都针对频繁集过滤一遍，并且按照降序排列
        local_data = {}
        for item in tran_set:
            if item in freq_item_set:
                # 如果是频繁集中的元素，就把频数赋给它
                local_data[item] = header_table[item][0]
        # local_data 形如 {'x': 3, 'z': 3} ，接下来还要按照"频数"降序排序

        if len(local_data) > 0:
            ordered_items = [v[0] for v in sorted(local_data.items(), key=lambda p: p[1], reverse=True)]

            # 每一条事务都要添加到树中，这个操作叫 update_tree

            update_tree(ordered_items, ret_tree, header_table, count)
    return ret_tree, header_table


def update_tree(ordered_items, in_tree, header_table, count):
    if ordered_items[0] in in_tree.children:
        # 如果开头的这个结点就在当前树结点的子结点里面，就给这个结点的频数增加
        in_tree.children[ordered_items[0]].inc(count)
    else:

        # 不在就马上创建一个新的结点挂上去，指明父节点
        in_tree.children[ordered_items[0]] = TreeNode(ordered_items[0], count, in_tree)

        # 再到头指针列表里面去查，如果头指针列表还没有指向，就要添加
        if header_table[ordered_items[0]][1] is None:
            header_table[ordered_items[0]][1] = in_tree.children[ordered_items[0]]
        else:
            # 如果有了，就去更新，让头指针列表的最后一个结点能够指向当前新的叶子结点
            # 更新头指针列表

            cur_node = header_table[ordered_items[0]][1]
            while cur_node.next_node:
                cur_node = cur_node.next_node
            cur_node.next_node = in_tree.children[ordered_items[0]]

    if len(ordered_items) > 1:
        # 递归调用
        update_tree(ordered_items[1:], in_tree.children[ordered_items[0]], header_table, count)


############ 以上，FP 树就建立好了 #############

# 挖掘频繁项集
def ascend_tree(leaf_node, prefix_path):
    if leaf_node.parent:
        prefix_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent, prefix_path)


def find_prefix_path(base_pat, tree_node):
    """
    爬树方法
    :param base_pat: 基模式，也就是叶子是是啥，根据 tree_node 回溯到根结点
    :param tree_node:
    :return:
    """
    # 条件模式，因为是在 base_pat 的前提下，所以也叫条件模式
    cond_pats = {}
    while tree_node:
        # 每一个头指针链表的节点都可以爬到一个路径
        prefix_path = []

        # 这个方法用于爬树
        # ascend_tree(tree_node, prefix_path)
        # 也可以使用下面的循环代替，声明一个新变量用于迭代

        cur_node = tree_node
        while cur_node.parent:
            prefix_path.append(cur_node.name)
            cur_node = cur_node.parent

        # print('基模式', base_pat, '前缀路径', prefix_path)
        # 只要路径大于 1 的，因为要把自己排除掉

        if len(prefix_path) > 1:
            # 得到前缀路径，前缀路径不包括基，它的 value 是基对应的 value
            cond_pats[frozenset(prefix_path[1:])] = tree_node.count

        tree_node = tree_node.next_node
    return cond_pats


def mine_tree(in_tree, header_table, min_sup, pre_fix, freq_item_list):
    # 这里将头结点指针列表按照频数升序排序
    # 【这一点很关键，因为我们要从下到上递归搜索得到频繁集列表】

    big_l = [v[0] for v in sorted(header_table.items(), key=lambda p: str(p[1][0]))]

    # print('big_l', big_l)

    for base_pat in big_l:

        # base_pat 是单个元素的频繁集
        # 前缀先从集合的开始
        new_freq_set = pre_fix.copy()

        # 【关键】前缀在这里增加
        new_freq_set.add(base_pat)

        # 因为是头指针列表的元素，所以一定是频繁集，先把它添加到频繁集列表中
        # 频繁模式在这里增加
        freq_item_list.append(new_freq_set)

        # 从头指针列表的链接开始，去爬树，
        # 根据每一个链接都能爬到一个前缀路径
        cond_path_bases = find_prefix_path(base_pat, header_table[base_pat][1])

        # print('base_pat', base_pat, '条件模式基：', cond_path_bases)

        # 把条件模式基（即向上爬得到的路径，又拿来创建树，会把不符合最小支持度要求的结点去掉）
        my_cond_tree, my_head_table = create_tree(cond_path_bases, min_sup)
        if my_head_table:
            # print('conditional tree for:', new_freq_set)
            mine_tree(my_cond_tree, my_head_table, min_sup, new_freq_set, freq_item_list)


# 生成数据集
def load_simple_data():
    simDat = [
        ['r', 'z', 'h', 'j', 'p'],
        ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
        ['z'],
        ['r', 'x', 'n', 'o', 's'],
        ['y', 'r', 'x', 'z', 'q', 't', 'p'],
        ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']
    ]
    return simDat


def create_init_set(data_set):
    ret_dict = {}
    for trans in data_set:
        ret_dict[frozenset(trans)] = 1
    return ret_dict


if __name__ == '__main__':
    # 设置最小支持度为 3
    min_support = 3
    simple_data = load_simple_data()

    print('加载的数据：')
    for transaction in simple_data:
        print(transaction)

    # 还要把数据处理一下，再送入算法
    init_set = create_init_set(simple_data)

    my_FP_tree, my_header_tab = create_tree(init_set, min_support)
    my_FP_tree.display()

    print(my_header_tab)
    # my_header_tab 是一个 dict
    # key 是单个元素频繁集，value 是 [频数，链表头结点]

    # 把找到的频繁集都放在这里
    my_freq_list = []
    # 前缀先从集合的开始
    mine_tree(my_FP_tree, my_header_tab, min_support, set([]), my_freq_list)
    print(len(my_freq_list))

