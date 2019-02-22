# 参考资料：https://time.geekbang.org/column/article/82943

from efficient_apriori import apriori
import csv
director = '宁浩'
file_name = './'+director+'.csv'
lists = csv.reader(open(file_name, 'r', encoding='utf-8-sig'))

print(lists)
# 数据加载
data = []
for names in lists:
     name_new = []
     for name in names:
           # 去掉演员数据中的空格
           name_new.append(name.strip())
     data.append(name_new[1:])
# 挖掘频繁项集和关联规则
itemsets, rules = apriori(data, min_support=0.5,  min_confidence=1)
print(itemsets)
print(rules)

# 从运行结果可以看出，宁浩导演比较爱用徐峥，并且有徐峥的电影，一定会用黄渤。

# 可以做一个爬春晚的爬虫
