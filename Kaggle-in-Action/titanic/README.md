# Titanic: Machine Learning from Disaster

[Kaggle](https://www.kaggle.com/) 是机器学习竞赛、托管数据库、编写和分享代码的平台，只要数据分析、机器学习、深度学习的从业者就一定会知道 Kaggle。在学习机器学习算法的时候，为了理解算法，往往我们会“手工”制造一些数据，以验证、研究算法的有效性。

Kaggle 上的竞赛和数据来自科研机构和大型企业，随着 Kaggle 的发展，全世界的数据科学爱好者和专家越来越多地聚集在这里，大型企业和科研机构会逐渐信赖这些参赛者，放心地提供一些数据，当然很多时候是脱敏的。这对于很多机器学习初学者来说，是一件非常好的事情，自己所学的知识终于有了用武之地。

国内类似 Kaggle 的平台有[天池](https://tianchi.aliyun.com/home/)、[科赛](https://www.kesci.com/)。而且这些使用这些平台还有一点好处，那就是**可以使用这些平台提供的开发环境，说白了就是“薅羊毛”，蹭 GPU 资源**，这一点对于新手来说，是大大的福利。

[“泰坦尼克号幸存者预测”](https://www.kaggle.com/c/titanic)是 Kaggle 上著名的初学者练习赛，是一个二分类问题，长年对用户开放，是数据科学小白的“Hello World”。任务很简单，提供了泰坦尼克当时船员的数据，一共包含 891 个训练样本和 418 个测试样本，要求参赛者使用 891 个样本训练模型，以预测 418 个测试样本幸存与否，其中 1 表示幸存（Survived），0 表示罹。

本文完整的源代码可以在我的 [GitHub]([https://github.com/liweiwei1419/Machine-Learning-is-Fun/blob/master/Kaggle-in-Action/titanic/notebook/Random-Forest-in-Titanic-Kaggle-Challenge.ipynb](https://github.com/liweiwei1419/Machine-Learning-is-Fun/blob/master/Kaggle-in-Action/titanic/notebook/Random-Forest-in-Titanic-Kaggle-Challenge.ipynb)
) 上查看，notebook 渲染慢的话，还可以在 [nbviewer]([https://nbviewer.jupyter.org/github/liweiwei1419/Machine-Learning-is-Fun/blob/master/Kaggle-in-Action/titanic/notebook/Random-Forest-in-Titanic-Kaggle-Challenge.ipynb](https://nbviewer.jupyter.org/github/liweiwei1419/Machine-Learning-is-Fun/blob/master/Kaggle-in-Action/titanic/notebook/Random-Forest-in-Titanic-Kaggle-Challenge.ipynb)
) 里查看，下面只是展示了部分关键思路，欢迎大家批评与指正。

下面是关于整个项目的工作流程。

## 工作流程

![泰坦尼克号幸存者预测.png](https://upload-images.jianshu.io/upload_images/414598-d2bd9d7f09fc1497.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 1、分析需求

![image.png](https://upload-images.jianshu.io/upload_images/414598-cfb0174b5dfd3d5e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这一步我们要做的是下载数据，并且熟悉各个字段的含义，知道目标变量的含义，预测任务是分类还是回归。如果字段不多，并且没有脱敏处理的话，不妨列一个表格，以便于我们更深刻地理解问题。

样本的各个特征以及含义如下：

| 序号 | 字段名      | 中文解释                             | 类型   | 说明                 |
| ---- | ----------- | ------------------------------------ | ------ | -------------------- |
| 1    | PassengerId | 乘客 ID                              | 离散型 |                      |
| 2    | Pclass      | 乘客等级                             | 离散型 | 1、2、3 等舱位、有序 |
| 3    | Name        | 乘客姓名                             | 离散型 |                      |
| 4    | Sex         | 性别                                 | 离散型 | female 和 male       |
| 5    | Age         | 年龄                                 | 连续型 | 整型                 |
| 6    | SibSp       | 与乘客一起旅行的兄弟姐妹和配偶的数量 | 连续型 | 整型                 |
| 7    | Parch       | 与乘客一起旅行的父母和孩子的数量     | 连续型 | 整型                 |
| 8    | Ticket      | 船票号码                             | 离散型 | 是一串文字           |
| 9    | Fare        | 票价                                 | 连续型 | 浮点数               |
| 10   | Cabin       | 客舱                                 | 离散型 | 缺失值较多           |
| 11   | Embarked    | 登船港口                             | 离散型 |                      |

这一步我们还要关注的一点是**评价指标**。评价指标往往关系着我们在使用**网格搜索**的时候最优超参数的选择。

先把下面的代码写上：
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
```

然后看看数据长什么样。

![image.png](https://upload-images.jianshu.io/upload_images/414598-a289f7c22416a9b0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下面的方法也是很常用的，看一看数据的数量、类型以及缺失值。

![image.png](https://upload-images.jianshu.io/upload_images/414598-c5e813e6e701c827.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

`DataFrame` 的 `describe()` 方法可以快速预览一些连续型变量的统计量。

![image.png](https://upload-images.jianshu.io/upload_images/414598-e6c1dbae18341067.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

数据的概览基本就到这里了，下面就要做一些数据可视化，进而帮助我们理解数据。

### 2、探索性数据分析

![image.png](https://upload-images.jianshu.io/upload_images/414598-7c1fcbf3a95ee426.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这一步我们分析各个特征对于目标变量 Survived 的影响。对于特征而言，我们首先关注它是离散型变量还是连续型变量。

+ 离散型变量：因为目标变量也是离散型变量，因此还可以使用卡方检验检测两个变量之间是否独立以验证可视化的结果。

这里以 Sex 变量为例。

![image.png](https://upload-images.jianshu.io/upload_images/414598-f87d96a35ca2f4b9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从图中可以看出，男士遇难的人数远超过女士，这也符合“泰坦尼克号”电影中船长决定女士和小孩优先上救生艇的事实。从统计学角度来看，就可以使用“卡方检验”来验证这个结论，首先我们得到列联表。

![image.png](https://upload-images.jianshu.io/upload_images/414598-fe37f82b000d1bf1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

进而计算卡方分布的 p 值，此时自由度为 $1$。

![image.png](https://upload-images.jianshu.io/upload_images/414598-4a7a33368fa8d7c4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看到 $p$ 值为 $1.1973570627755645e-58$，远小于 $0.05$，因此“性别”和“幸存”的确不是独立的，“性别”是一项预测“是否幸存”的重要特征。

关于卡方检验的知识可以查看我的文章 [白话“卡方检验”](https://www.liwei.party/2019/02/11/machine-learning/chi-square-test/)。


+ 连续型变量：根据目标变量分别绘制频率分布直方图。

（1）Age 

![image.png](https://upload-images.jianshu.io/upload_images/414598-d57916c63738dc19.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看出，年轻人幸存较多。

（2）Fare

![image.png](https://upload-images.jianshu.io/upload_images/414598-ba49f12b15d84489.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看出，票价比较低的，遇难的人数比较多，因此票价是一个重要的特征。

### 3、特征工程

![image.png](https://upload-images.jianshu.io/upload_images/414598-0292cf8fadff320f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


其实以上等于什么都没有做，数据还是原始数据。接下来合并训练数据和测试数据，分离出目标变量和测试数据的 ID，具体如下。

![image.png](https://upload-images.jianshu.io/upload_images/414598-10b3cb14eade824b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


下面就得“动真格”了，主要干的事情有：

（1）缺失值填充；
连续型变量可以用平均数，离散型变量用众数，或者分组填充，还可以凭借领域知识填充，更有甚者使用监督学习的方法预测缺失值。

本例中 Age 变量的缺失值填充根据 Sex、Pclass 和 Title 分组，如果落在相同的组别里，就用这个组别的平均数填充。

![image.png](https://upload-images.jianshu.io/upload_images/414598-6d4ad160939cf415.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/414598-f30b700a04469d77.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

（2）特征抽取

如果数据是文本类型的，一般要做特征抽取，因为绝大多数机器学习算法是不能直接输入文本的。例如本例中可以根据 Name 字段抽取每个人的社会地位，具体代码如下：

![image.png](https://upload-images.jianshu.io/upload_images/414598-7d74343adb849dbb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/414598-d1b406e52a085cbe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/414598-1c09c119dae250d6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

上面的代码可以简单总结为“合并同类项”。接着就要进行独热编码了，即将多分类变量转换为二分类变量。这一步我们放在最后和其它离散型变量一起进行独热编码。

还可以抽取的变量有“家庭人数”。

![image.png](https://upload-images.jianshu.io/upload_images/414598-ed9376ea44ffe8cd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


（3）连续型变量分箱处理成离散型变量

如果连续型变量本身与目标变量之间不是线性关系，那么有时候为了简化问题，可以考虑将其进行分箱处理成离散型变量。

这里我们对“Age”和“家庭人数”进行分箱处理。

![image.png](https://upload-images.jianshu.io/upload_images/414598-06d2953e91a21af2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/414598-bd962a37f8cb117b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

最后做离散型变量的独热编码，并且分离出训练数据集和测试数据集：

![image.png](https://upload-images.jianshu.io/upload_images/414598-0efe2ca5919323bd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们看一眼数据：

![image.png](https://upload-images.jianshu.io/upload_images/414598-88901cb0d1d178bc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


到目前为止训练数据集和测试数据集都是“规规整整”的数值了，可以送入机器学习算法了。

### 4、模型训练与评估

![image.png](https://upload-images.jianshu.io/upload_images/414598-312256c89fddfb20.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

（1）使用集成学习方法

现在的 Kaggle 平台上面的任务几乎都不会使用单模型，我们使用集成学习中的随机森林模型。它的优点如下：

+ 对离群点不敏感；
+ 随机森林的泛化误差随着树的增多而收敛，不容易过拟合；
+ 每棵树在划分的时候考虑的候选特征较少，计算速度快；
+ 可以给出特征重要性的估计；
+ 可以并行计算。

当然，我们还可以使用大名鼎鼎的 XGBoost。

（2）网格搜索

网格搜索是提高算法有效性的重要手段，得凭一些经验，多进行尝试。**由于网格搜索比较耗时，因此可以设置一个开关，把网格搜索的结果写在另一个分支里，重新执行代码的时候就不用重新训练了，直接使用最佳超参数就好**。

```python
%%time
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# 网格搜索开关
run_gs = False

if run_gs:
    parameter_grid = {
        'max_depth': [4, 6, 8],
        'n_estimators': [50, 10],
        'max_features': ['sqrt', 'auto', 'log2'],
        'min_samples_split': [2, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'bootstrap': [True, False],
    }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(
        forest,
        scoring='accuracy',
        param_grid=parameter_grid,
        cv=cross_validation,
        verbose=1)

    grid_search.fit(train_reduced, y)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

else:
    parameters = {
        'bootstrap': False,
        'min_samples_leaf': 1,
        'n_estimators': 10,
        'min_samples_split': 3,
        'max_features': 'log2',
        'max_depth': 8
    }
```

```python
model = RandomForestClassifier(**parameters)
model.fit(X_train, y)
```

![image.png](https://upload-images.jianshu.io/upload_images/414598-1b1280f28de1ee54.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### 5、预测并提交

```python
y_pred = model.predict(X_test)
res = pd.DataFrame({'PassengerId': PassengerId, 'Survived': y_pred})
res.to_csv('../output/gridsearch_rf_2019_05_27.csv', index=False)
```
提交结果：

![随机森林+网格搜索的提交结果](https://upload-images.jianshu.io/upload_images/414598-2136ff63ba3dc0a7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

此外，还要还看到，在训练数据集上的得分是：$0.815$，在测试数据集上的得分是：$0.804$，说明泛化性能还不错。另外，使用未调参的 XGBoost 也得到了同样的准确率，同样可以进行网格搜索，因为我对 XGBoost 的参数不太熟悉，在这里就略过了，请大家指教。

![使用 XGBoost 未调参也得到了同样的准确率](https://upload-images.jianshu.io/upload_images/414598-564dc347fec87f9b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 总结

+ 在 Leaderboard 上，有人公开了得分为 $1.0$ 的[方案](https://www.kaggle.com/tarunpaparaju/how-top-lb-got-their-score-use-titanic-to-learn)，告诉我们其实有些得分是“作弊”来的，我们泰坦尼克号幸存者问题其实不太适合用作模型训练，得分在 $0.8$ 上下就已经可以了，没有必要再继续优化，不必在上面多花时间；
+ 多在 Kaggle 的 Kernels 和 Discussion 区查看别人提供的代码，说不定可以为自己今后要解决的问题提供思路；
+ 我还试过使用神经网络训练，但是泛化性能较差，这是因为数据太少。