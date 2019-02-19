# 封装一下 FP-growth 算法


class FPGrowth:

    def __init__(self, min_support):
        self.min_support = min_support
        self.my_freq_list_ = []

    def fit(self, data_set):
        data_set = self.__create_init_set(data_set)
        my_header_tab = self.__create_tree(data_set)
        self.mine_tree(my_header_tab, set([]))

    def __create_tree(self, data_set):
        header_table = {}
        for trans in data_set:
            for item in trans:
                header_table[item] = header_table.get(item, 0) + data_set[trans]
        header_table = dict(filter(lambda x: x[1] >= self.min_support, header_table.items()))

        freq_item_set = set(header_table.keys())
        if len(freq_item_set) == 0:
            return None

        for k in header_table:
            header_table[k] = [header_table[k], None]

        ret_tree = FPGrowth.TreeNode('Null Set', 0, None)

        for tran_set, count in data_set.items():
            local_data = {}
            for item in tran_set:
                if item in freq_item_set:
                    local_data[item] = header_table[item][0]

            if len(local_data) > 0:
                ordered_items = [v[0] for v in sorted(local_data.items(), key=lambda p: p[1], reverse=True)]

                self.__update_tree(ordered_items, ret_tree, header_table, count)
        return header_table

    def __update_tree(self, ordered_items, in_tree, header_table, count):
        if ordered_items[0] in in_tree.children:
            in_tree.children[ordered_items[0]].inc(count)
        else:
            in_tree.children[ordered_items[0]] = FPGrowth.TreeNode(ordered_items[0], count, in_tree)

            if header_table[ordered_items[0]][1] is None:
                header_table[ordered_items[0]][1] = in_tree.children[ordered_items[0]]
            else:

                cur_node = header_table[ordered_items[0]][1]
                while cur_node.next_node:
                    cur_node = cur_node.next_node
                cur_node.next_node = in_tree.children[ordered_items[0]]

        if len(ordered_items) > 1:
            self.__update_tree(ordered_items[1:], in_tree.children[ordered_items[0]], header_table, count)

    def __create_init_set(self, data_set):
        ret_dict = {}
        for trans in data_set:
            ret_dict[frozenset(trans)] = 1
        return ret_dict

    def find_prefix_path(self, tree_node):
        cond_pats = {}
        while tree_node:
            prefix_path = []

            cur_node = tree_node
            while cur_node.parent:
                prefix_path.append(cur_node.name)
                cur_node = cur_node.parent

            if len(prefix_path) > 1:
                cond_pats[frozenset(prefix_path[1:])] = tree_node.count

            tree_node = tree_node.next_node
        return cond_pats

    def mine_tree(self, header_table, pre_fix):
        big_l = [v[0] for v in sorted(header_table.items(), key=lambda p: str(p[1][0]))]

        for base_pat in big_l:

            new_freq_set = pre_fix.copy()

            new_freq_set.add(base_pat)

            self.my_freq_list_.append(new_freq_set)
            cond_path_bases = self.find_prefix_path(header_table[base_pat][1])

            my_head_table = self.__create_tree(cond_path_bases)
            if my_head_table:
                self.mine_tree(my_head_table, new_freq_set)

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


if __name__ == '__main__':
    fpg = FPGrowth(min_support=3)

    simp_dat = [['r', 'z', 'h', 'j', 'p'],
                ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                ['z'],
                ['r', 'x', 'n', 'o', 's'],
                ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    fpg.fit(simp_dat)

    print('找到的频繁项集：')
    for index, frequent_item_sets in enumerate(fpg.my_freq_list_):
        print(index, frequent_item_sets)

    print('------')
    data_set = [
        [1, 3, 4],
        [2, 3, 5],
        [1, 2, 3, 5],
        [2, 5]
    ]
    fpg = FPGrowth(min_support=2)
    fpg.fit(data_set)
    for index, frequent_item_sets in enumerate(fpg.my_freq_list_):
        print(index, frequent_item_sets)

    # 得到的频繁集列表如下：
    # [frozenset({1}), frozenset({3}), frozenset({2}), frozenset({5})]
    # [frozenset({1, 3}), frozenset({2, 3}), frozenset({3, 5}), frozenset({2, 5})]
    # [frozenset({2, 3, 5})]
