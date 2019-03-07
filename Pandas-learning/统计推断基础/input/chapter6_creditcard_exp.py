# coding: utf-8

# # 第6讲 统计推断基础
# - 数据说明：本数据是地区房价增长率数据
# - 名称-中文含义
# - dis_name-小区名称
# - rate-房价同比增长率
# %%

import os

os.chdir(r"D:\Python_book\6Inference")
# In[1]:

import pandas as pd

house_price_gr = pd.read_csv(r'house_price_gr.csv', encoding='gbk')
house_price_gr.head()

# ## 6.1 参数估计
# 进行描述性统计分析

# In[2]:

house_price_gr.describe(include='all')

# Histograph

# In[3]:

get_ipython().magic('matplotlib inline')
import seaborn as sns
from scipy import stats

sns.distplot(house_price_gr.rate, kde=True, fit=stats.norm)  # Histograph

# Q-Q

# In[4]:

import statsmodels.api as sm
from matplotlib import pyplot as plt

fig = sm.qqplot(house_price_gr.rate, fit=True, line='45')
fig.show()

# Box Plots

# In[5]:

house_price_gr.plot(kind='box')  # Box Plots

# 置信度区间估计

# In[6]:

se = house_price_gr.rate.std() / len(house_price_gr) ** 0.5
LB = house_price_gr.rate.mean() - 1.98 * se
UB = house_price_gr.rate.mean() + 1.98 * se
(LB, UB)


# In[7]:

# 如果要求任意置信度下的置信区间的话，可以自己编一个函数
def confint(x, alpha=0.05):
    n = len(x)
    xb = x.mean()
    df = n - 1
    tmp = (x.std() / n ** 0.5) * stats.t.ppf(1 - alpha / 2, df)
    return {'Mean': xb, 'Degree of Freedom': df, 'LB': xb - tmp, 'UB': xb + tmp}


confint(house_price_gr.rate, 0.05)

# In[8]:

# 或者使用DescrStatsW
d1 = sm.stats.DescrStatsW(house_price_gr.rate)
d1.tconfint_mean(0.05)  #

# ## 6.2 假设检验与单样本T检验
# 当年住宅价格的增长率是否超过了10%的阈值

# In[9]:

d1 = sm.stats.DescrStatsW(house_price_gr.rate)
print('t-statistic=%6.4f, p-value=%6.4f, df=%s' % d1.ttest_mean(0.1))

# ## 6.3 两样本T检验
# 导入数据
# 数据说明：本数据是一份汽车贷款数据

# |字段名|中文含义|
# |:--:|:--:|
# |id|id|
# |Acc|是否开卡(1=已开通)|
# |avg_exp|月均信用卡支出（元）|
# |avg_exp_ln|月均信用卡支出的自然对数|
# |gender|性别(男=1)|
# |Age|年龄|
# |Income|年收入（万元）|
# |Ownrent|是否自有住房（有=1；无=0)|
# |Selfempl|是否自谋职业(1=yes, 0=no)|
# |dist_home_val|所住小区房屋均价(万元)|
# |dist_avg_income|当地人均收入|
# |high_avg|高出当地平均收入|
# |edu_class|教育等级：小学及以下开通=0，中学=1，本科=2，研究生=3|
# In[10]:

creditcard = pd.read_csv(r'creditcard_exp.csv', skipinitialspace=True)

# In[11]:

creditcard['Income'].groupby(creditcard['Acc']).describe()

# - 第一步:方差齐次检验

# In[12]:

Suc0 = creditcard[creditcard['Acc'] == 0]['Income']
Suc1 = creditcard[creditcard['Acc'] == 1]['Income']
leveneTestRes = stats.levene(Suc0, Suc1, center='median')
print('w-value=%6.4f, p-value=%6.4f' % leveneTestRes)
# - 第二步:T-test

# In[13]:
stats.stats.ttest_ind(Suc0, Suc1, equal_var=False)
# Or Try: sm.stats.ttest_ind(gender0, gender1, usevar='pooled')
# %%
# 测试一下性别对是月均消费的作用.
# 注意对缺失值得处理
# creditcard['avg_exp'].groupby(creditcard['gender']).describe()
# %%
# female= creditcard[creditcard['gender'] == 0]['avg_exp'].dropna()
# male = creditcard[creditcard['gender'] == 1]['avg_exp'].dropna()
# leveneTestRes = stats.levene(female, male, center='median')
# print('w-value=%6.4f, p-value=%6.4f' %leveneTestRes)
# %%
# stats.stats.ttest_ind(female, male, equal_var=True)

# ## 6.4 方差分析
# - 单因素方差分析

# In[14]:

pd.set_option('display.max_columns', None)  # 设置显示所有列
creditcard.groupby('edu_class')[['avg_exp']].describe().T

# In[15]:

# 利用回归模型中的方差分析
import statsmodels.api as sm
from statsmodels.formula.api import ols

sm.stats.anova_lm(ols('avg_exp ~ C(edu_class)', data=creditcard).fit())

# - 多因素方差分析

# In[16]:不考虑交互相

sm.stats.anova_lm(ols('avg_exp ~ C(edu_class)+C(gender)', data=creditcard).fit())
# In[16]:考虑交互相
sm.stats.anova_lm(ols('avg_exp ~ C(edu_class)+C(gender)+C(edu_class)*C(gender)', data=creditcard).fit())

# ## 6.5 相关分析
# 散点图

# In[ ]:

creditcard.plot(x='Income', y='avg_exp', kind='scatter')
# 当发现散点图有发散的趋势时，首先需要对Y取对数，而且还应该尝试对X也取对数
# %%
creditcard.plot(x='Income', y='avg_exp_ln', kind='scatter')
# 相关性分析:“spearman”,“pearson” 和 "kendall"
# In[ ]:
# import numpy as np
# creditcard['Income_ln']=np.log(creditcard['Income'])
# In[ ]:
creditcard[['avg_exp_ln', 'Income']].corr(method='pearson')

# ## 6.6卡方检验

# In[7]:

cross_table = pd.crosstab(creditcard.edu_class, columns=creditcard.Acc)
# Or try this: accepts.pivot_table(index='bankruptcy_ind',columns='bad_ind', values='application_id', aggfunc='count')
cross_table

# In[9]:

cross_table_rowpct = cross_table.div(cross_table.sum(1), axis=0)
cross_table_rowpct

# In[ ]:

print('chisq = %6.4f\n p-value = %6.4f\n dof = %i\n expected_freq = %s' % stats.chi2_contingency(cross_table))

# In[ ]:
