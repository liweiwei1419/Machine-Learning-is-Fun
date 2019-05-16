# Bike Sharing Demand

+ 自行车共享需求

+ 华盛顿自行车租赁数据

+ 竞赛地址：https://www.kaggle.com/c/bike-sharing-demand

+ 时间序列方法

+ 这是一个回归问题：均方根对数误差：RMSLE

  


自行车共享系统是租赁自行车的一种方式，其中通过整个城市的自助服务终端网络自动获得会员资格，租赁和自行车返回的过程。使用这些系统，人们可以从一个地方租用自行车，并根据需要将其返回到不同的地方。目前，全世界有超过 500 个自行车共享计划。

这些系统生成的数据使其对研究人员具有吸引力，因为明确记录了旅行的持续时间，出发地点，到达地点和经过的时间。因此，自行车共享系统用作传感器网络，其可用于研究城市中的移动性。在本次比赛中，参与者被要求将历史使用模式与天气数据相结合，以预测华盛顿特区 Capital Bikeshare 计划中的自行车租赁需求。

致谢 Acknowledgements


Kaggle 正在举办本次机器学习社区竞赛，以用于娱乐和练习。该数据集由 Hadi Fanaee Tork 使用 Capital Bikeshare 的数据提供。我们还要感谢UCI 机器学习库来托管数据集。如果您在出版物中使用该问题，请引用：

Fanaee-T，Hadi和Gama，Joao，结合集合探测器和背景知识的事件标记，人工智能进展（2013）：第1-15页，Springer Berlin Heidelberg。

提交的评估为均方根对数误差（RMSLE）。 RMSLE 计算为

$$
\sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 }
$$

## 提交格式

您的提交文件必须有标题，并且应按以下格式构建：

datetime,count
2011-01-20 00:00:00,0
2011-01-20 01:00:00,0
2011-01-20 02:00:00,0
...
...

See, fork, and run a random forest benchmark model through Kaggle Scripts

通过 Kaggle Scripts 查看，分叉并运行随机森林基准模型。

You are provided hourly rental data spanning two years. For this competition, the training set is comprised of the first 19 days of each month, while the test set is the 20th to the end of the month. You must predict the total count of bikes rented during each hour covered by the test set, using only information available prior to the rental period.

您将获得两年的每小时租赁数据。对于本次比赛，训练集由每月的前19天组成，而测试组则是20月至月底。您必须使用租赁期之前可用的信息预测测试集所涵盖的每小时内租用的自行车总数。

## Data Fields 

数据字段

| 字段                                                         | 说明                                      |
| ------------------------------------------------------------ | ----------------------------------------- |
| datetime - hourly date + timestamp                           | 租赁时间- 每小时日期+时间戳               |
| season -  1 = spring, 2 = summer, 3 = fall, 4 = winter       | 季节-  1 =春天，2 =夏天，3 =秋天，4 =冬天 |
| holiday - whether the day is considered a holiday            | 假期 - 这一天是否为假日，0=非假日，1=假日 |
| workingday - whether the day is neither a weekend nor holiday | 工作日 - 这一天既不是周末也不是假期       |
| weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy <br/>2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist <br/>3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds <br/>4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog | 天气情况，数字越大，天气越差<br>          |
| temp - temperature in Celsius                                | 温度 - 摄氏温度                           |
| atemp - "feels like" temperature in Celsius                  | atemp  - “感觉就像”摄氏温度               |
| humidity - relative humidity                                 | 湿度 - 相对湿度                           |
| windspeed - wind speed                                       | windspeed  - 风速                         |
| casual - number of non-registered user rentals initiated     | casual  - 已启动的非注册用户租赁数量      |
| registered - number of registered user rentals initiated     | registered  - 已启动的注册用户租赁数量    |
| count - number of total rentals                              | count  - 总租金数量                       |

weather：天气 -  1：晴朗，少云，晴间多云，多云<br/>2：雾+多云，薄雾+破云，雾+少云，雾<br/>3：小雪，小雨+雷暴+散云，小雨+散云<br/>4：暴雨+冰托+雷暴+雾，雪+雾



+ weekday 更具有周期性、代表性。



参考资料：

1、https://www.cnblogs.com/en-heng/p/6907839.html

2、scikit-learn 模型评价指标：msle

https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-log-error
$$
\text{MSLE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (\log_e (1 + y_i) - \log_e (1 + \hat{y}_i) )^2.
$$


华盛顿自行车问题的评价指标是：RMSLE。
$$
\sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 }
$$
参考资料：https://www.cnblogs.com/jiaxin359/p/8989565.html





## 思路整理

1、抽取特征：

年、月、日、小时、星期几、日期

2、分离训练数据与测试数据

```{.python .input}
train = data.iloc[:10886]
test = data.iloc[10886:]
```

3、正相关、负相关

可以取绝对值，负相关也是相关。

4、EDA 的一些结论

+ 2012 年的租借数明显比 2011 年高，说明随着时间的推移，共享单车逐渐被更多的人熟悉和认可，使用者越来越多。

+ 月份对租借数影响显著，从 1 月份开始每月的租借数快速增加，到 6 月份达到顶峰，随后至 10 月缓慢降低，10月后急剧减少。这明显与季节有关。

+ 通过各季度箱型图可以看出季节对租借数的影响符合预期：春季天气仍然寒冷，骑车人少；随着天气转暖，骑车人逐渐增多，并在秋季（天气最适宜时）达到顶峰；随后进入冬季，天气变冷，骑车人减少。

  **因为月份和季节对租借数的影响重合，且月份更加详细，因此在随后的建模过程中选取月份特征，删除季节特征。**

+ 从时间的分布上来看，每天有两个高峰期，分别是早上 8 点左右和下午 17 点左右，正好是工作日的上下班高峰期。而介于两者之间的白天时间变化规律不明显，可能与节假日有关，因此以此为基础需要考虑节假日和星期的影响。



可以看出，工作日早晚上班高峰期租借量高，其余时间租借量低；节假日中午及午后租借量较高，符合节假日人们出行用车的规律。
