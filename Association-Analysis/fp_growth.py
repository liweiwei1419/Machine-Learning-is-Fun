from collections import defaultdict

# 测试用，已经坏了

class TreeNode:
    def __init__(self, name, count, parent_node):
        # 结点名称
        self.name = name
        # 出现次数
        self.count = count

        # 指向下一个相似结点的指针，默认为None
        self.node_link = None

        # 指向父结点的指针，在构造时初始化为给定值

        self.parent_node = parent_node

        # children：指向子结点的字典，
        # 以子结点的元素名称为键，指向子结点的指针为值，
        # 初始化为空字典
        self.children_nodes = {}

    def inc(self, num_occur):
        # 增加结点的出现次数值
        self.count += num_occur

    def disp(self, deep=1):
        # 输出结点和子结点的 FP 树结构
        print(' ' * deep, self.name, self.count)
        for child in self.children_nodes.values():
            child.display(deep + 1)


if __name__ == '__main__':
    root = TreeNode('pyramid', 9, None)
    root.children_nodes['eye'] = TreeNode('eye', 13, None)
    root.children_nodes['phoenix'] = TreeNode('phoenix', 3, None)

    root.disp()

    data_set = [['r', 'z', 'h', 'j', 'p'],
                ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                ['z'],
                ['r', 'x', 'n', 'o', 's'],
                ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]

    header_table = defaultdict(int)

    for trans in data_set:
        for item in trans:
            header_table[item] += 1
    # 不就是记个数吗，整的这么麻烦，真的是！！！
    for k, v in header_table.items():
        print(k, v)

    # 这里最小支持度设置为 3
    min_support = 3

    # 移除不满足最小支持度的元素项
    header_table = dict(filter(lambda x: x[1] >= min_support, header_table.items()))
    print(header_table)

    freq_item_set = set(header_table.keys())

    if len(freq_item_set) == 0:
        pass
        # 返回空

    # 增加一个数据项，用于存放指向相似元素项指针
    for k in header_table:
        header_table[k] = [header_table[k], None]

    for k, v in header_table.items():
        print(k, v)

    # r [3, None]
    # z [5, None]
    # y [3, None]
    # x [4, None]
    # t [3, None]
    # s [3, None]

    # 这是一棵 FP 树的根结点，根结点的名字，我们就让它是空好了
    FPTree = TreeNode('ROOT', 0, None)  # 根节点


    def updateTree(items, inTree, header_table, count):
        # item 长这样：
        # item['z', 'r']
        # item['z', 'x', 'y', 't', 's']
        # item['z']
        # item['x', 'r', 's']
        # item['z', 'x', 'y', 'r', 't']
        # item['z', 'x', 'y', 's', 't']

        if items[0] in inTree.children_nodes:
            inTree.children_nodes[items[0]].inc(count)

        else:
            # 没有的时候，创建一个结点
            inTree.children_nodes[items[0]] = TreeNode(name=items[0], count=count, parent_node=inTree)

            # 如果在索引表中结点还没有链接指向的时候，就指定一个节点指向
            if header_table[items[0]][1] is None:
                header_table[items[0]][1] = inTree.children_nodes[items[0]]
            else:

                # 如果 r [3, None] 在 Node 位置有结点指向的话
                # 看看 node_link 是否为空，如果非空，就滑到那个空的地方去
                while header_table[items[0]][1].node_link:
                    header_table[items[0]][1] = header_table[items[0]][1].node_link
                # 直到 header_table[items[0]][1] 这个结点的 node_link 为空
                # 就把 inTree.children[items[0]] 指向它
                header_table[items[0]][1].node_link = inTree.children_nodes[items[0]]

        if len(items) > 1:
            # 对剩下的元素项递归调用updateTree函数
            updateTree(items[1::], inTree.children_nodes[items[0]], header_table, count)


    # 第二次遍历数据集，创建FP树
    for tranSet in data_set:
        localD = {}  # 对一个项集tranSet，记录其中每个元素项的全局频率，用于排序
        for item in tranSet:
            if item in freq_item_set:
                localD[item] = header_table[item][0]  # 注意这个[0]，因为之前加过一个数据项
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]  # 排序
            print('item', orderedItems)

            # 根据一条交易记录更新一棵树
            updateTree(orderedItems, FPTree, header_table, 1)  # 更新FP树

    FPTree.disp()

    # 就是图上画的那个样子
    print(header_table)


    # 树画出来了，下面就要挖掘频繁项集了

    # 1、得到每个元素的前缀路径

    def ascendTree(leafNode, prefixPath):
        print(leafNode)
        if leafNode.parent_node:
            prefixPath.append(leafNode.name)
            ascendTree(leafNode.parent_node, prefixPath)


    def find_pre_fix_path(base_path, treeNode):
        cond_paths = {}
        while FPTree:
            prefix_path = []
            ascendTree(treeNode, prefix_path)
            if len(prefix_path) > 1:
                cond_paths[frozenset(prefix_path[1:])] = treeNode.count
            treeNode = treeNode.node_link
        return cond_paths

    print('11',header_table['x'][1])

    print(find_pre_fix_path('x', header_table['x'][1]))
