# 旧金山犯罪预测

地址：https://www.kaggle.com/c/sf-crime/overview

评价指标：
$$
log loss = -\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^My_{ij}\log(p_{ij}),
$$

| 特征名     | 解释                              |
| ---------- | --------------------------------- |
| Date       | 日期                              |
| Category   | 犯罪类型，比如 Larceny 盗窃罪 等 |
| Descript   | 对于犯罪更详细的描述              |
| DayOfWeek  | 星期几                            |
| PdDistrict | 所属警区                          |
| Resolution | 处理结果                          |
| Address    | 发生街区位置                      |
| X          | 经度                              |
| Y          | 纬度                              |

参考资料：

https://blog.csdn.net/han_xiaoyang/article/details/50629608