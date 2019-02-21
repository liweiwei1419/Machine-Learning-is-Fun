import matplotlib.pyplot as plt



############### 以下代码用于绘图 #########################
###################### 绘图 #######################
# 计算叶子节点的个数
# 叶子节点的个数，就是画出的图的宽度
def get_num_leafs(decision_tree_json):
    num_leafs = 0
    root = list(decision_tree_json.keys())[0]
    second_dict = decision_tree_json[root]
    for key, value in second_dict.items():
        if type(value).__name__ == 'dict':
            num_leafs += get_num_leafs(value)
        else:
            num_leafs += 1
    return num_leafs


# 计算树的层数
# 树的层数其实就是画出的图的高度
def get_tree_depth(decision_tree_json):
    max_depth = 0
    root = list(decision_tree_json.keys())[0]
    second_dict = decision_tree_json[root]
    for key, value in second_dict.items():
        if type(value).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(value)
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth



decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def create_plot(in_tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plot_tree.totalW = float(get_num_leafs(in_tree))
    plot_tree.totalD = float(get_tree_depth(in_tree))
    plot_tree.xOff = -0.5 / plot_tree.totalW;
    plot_tree.yOff = 1.0;
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()


def plot_node(node_txt, center_point, parent_point, node_type):
    create_plot.ax1.annotate(node_txt, xy=parent_point, xycoords='axes fraction', xytext=center_point,
                             textcoords='axes fraction', va='center', ha='center', bbox=node_type,
                             arrowprops=arrow_args)


def plot_mid_text(cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30)


def plot_tree(my_tree, parent_pt, node_txt):  # if the first key tells you what feat was split on
    num_leafs = get_num_leafs(my_tree)  # this determines the x width of this tree
    depth = get_tree_depth(my_tree)
    first_str = list(my_tree.keys())[0]  # the text label for this node should be this
    cntr_pt = (plot_tree.xOff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    second_dict = my_tree[first_str]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in second_dict.keys():
        if type(second_dict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plot_tree(second_dict[key], cntr_pt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntr_pt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD

# if you do get a dictonary you know it's a tree, and the first element will be another dict




if __name__ == '__main__':
    pass



