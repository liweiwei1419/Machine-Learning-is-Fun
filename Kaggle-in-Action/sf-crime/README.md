# San Francisco Crime Classification

地址：[https://www.kaggle.com/c/sf-crime/overview](https://www.kaggle.com/c/sf-crime/overview)

本文完整的源代码可以在我的 [GitHub](https://github.com/liweiwei1419/Machine-Learning-is-Fun/blob/master/Kaggle-in-Action/sf-crime/notebook/MultinomialNB-sf-crime-rate-prediction.ipynb) 上查看，notebook 渲染慢的话，还可以在 [nbviewer]([https://nbviewer.jupyter.org/github/liweiwei1419/Machine-Learning-is-Fun/blob/master/Kaggle-in-Action/sf-crime/notebook/MultinomialNB-sf-crime-rate-prediction.ipynb) 里查看，下面只是展示了部分关键思路，欢迎大家批评与指正。

下面是关于整个项目的工作流程。

## 工作流程

![旧金山犯罪分类预测](https://upload-images.jianshu.io/upload_images/414598-1604c664fc5608a6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### 1、分析需求

“旧金山犯罪分类预测”是一个多分类问题，要求我们预测犯罪类型。使用的评价指标是 logloss：

$$
log loss = -\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^My_{ij}\log(p_{ij}),
$$

实际上就是交叉熵损失函数，这个值越小，就说明算法的预测效果越好。并且要求我们的算法能够预测概率，多分类问题可以预测概率的模型有使用决策树、随机森林、 朴素贝叶斯等。

样本的各个特征以及含义如下：

| 序号   | 特征名     | 解释                              |
| ---------- | --------------------------------- | --------------------------------- |
| 1      | Date       | 日期                              |
| 2  | Category   | 犯罪类型，比如 Larceny 盗窃罪 等 |
| 3  | Descript   | 对于犯罪更详细的描述              |
| 4 | DayOfWeek  | 星期几                            |
| 5 | PdDistrict | 所属警区                          |
| 6 | Resolution | 处理结果                          |
| 7   | Address    | 发生街区位置                      |
| 8         | X          | 经度                              |
| 0         | Y          | 纬度                              |

拿到数据以后，我们看一看数量、特征个数，是否有缺失值。

![kaggle-in-action-sf-crime-1](https://upload-images.jianshu.io/upload_images/414598-3ad6ee6f17cf7230.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![kaggle-in-action-sf-crime-2](https://upload-images.jianshu.io/upload_images/414598-4be3583fa106bd2e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看到，数据没有缺失值，训练数据集有 $87.8$ 万，测试数据集有 $88.4$ 万。由于这里数据量很大，我们使用朴素贝叶斯算法。

下面的代码为提交做准备：

![kaggle-in-action-sf-crime-3](https://upload-images.jianshu.io/upload_images/414598-f3063224471cee6c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 2、探索性数据分析

![kaggle-in-action-sf-crime-4](https://upload-images.jianshu.io/upload_images/414598-5956876e6d64636b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看出周五犯罪的案件最多。注意到特征中有街区信息和经纬度，其实可以做出[很漂亮的可视化效果]([https://www.kaggle.com/abhimicro3/eda-of-san-francisco-crime-classification/notebook](https://www.kaggle.com/abhimicro3/eda-of-san-francisco-crime-classification/notebook)
)，不过鉴于我现有的知识，在这里就不做展示了。

### 3、特征工程

这里做的特征工程也比较简单，抽取年、月、日、小时、星期几、街区信息，然后再独热编码。

![kaggle-in-action-sf-crime-5](https://upload-images.jianshu.io/upload_images/414598-d7f3753f7fd6e31c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

以上是对训练数据集的处理，而测试数据集也进行相同的处理，这里不再赘述。

### 4、模型训练与评估

我们首先从训练数据集中分割出一部分作为验证集，看一看效果如何。

![kaggle-in-action-sf-crime-6](https://upload-images.jianshu.io/upload_images/414598-ba6663657bdb74d5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

得到对数损失函数的值为 $2.65$，说明还不错。然后我们使用全部的训练数据重新训练模型。

![kaggle-in-action-sf-crime-7](https://upload-images.jianshu.io/upload_images/414598-d3af54523c819e0b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### 5、预测并提交

![kaggle-in-action-sf-crime-8](https://upload-images.jianshu.io/upload_images/414598-c0b9a2d21bd75927.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![kaggle-in-action-sf-crime-9](https://upload-images.jianshu.io/upload_images/414598-f2d70c4d3aae3033.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在排行榜 $63\%$ 左右，成绩不是很理想。问题在于特征工程比较粗糙，并且模型其实选择了一个比较粗略的模型（为了兼顾训练速度和预测速度）。在 Kernels 中看到的比较优秀的特征工程抽取了“经纬度只差”与“经纬度之和”特征，并且使用了比 XGBoost 还快的 lightgbm，最后得分是 $2.2510$，可以[参考](https://github.com/liweiwei1419/Machine-Learning-is-Fun/blob/master/Kaggle-in-Action/sf-crime/notebook/lightgbm-sf-crime-rate-prediction-2.2510.ipynb)。

![kaggle-in-action-sf-crime-10](https://upload-images.jianshu.io/upload_images/414598-ff8fa16c32639823.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## 参考资料

1、https://blog.csdn.net/han_xiaoyang/article/details/50629608

