
# coding: utf-8

# # 第6讲 统计推断基础
# - 数据说明：本数据是地区房价增长率数据
# - 名称-中文含义
# - dis_name-小区名称
# - rate-房价同比增长率
#%%

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

sns.distplot(house_price_gr.rate, kde=True, fit=stats.norm) # Histograph


# Q-Q

# In[4]:

import statsmodels.api as sm
from matplotlib import pyplot as plt

fig = sm.qqplot(house_price_gr.rate, fit=True, line='45')
fig.show()


# Box Plots

# In[5]:

house_price_gr.plot(kind='box') # Box Plots


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
    df = n-1
    tmp = (x.std() / n ** 0.5) * stats.t.ppf(1-alpha/2, df)
    return {'Mean': xb, 'Degree of Freedom':df, 'LB':xb-tmp, 'UB':xb+tmp}

confint(house_price_gr.rate, 0.05)


# In[8]:

# 或者使用DescrStatsW
d1 = sm.stats.DescrStatsW(house_price_gr.rate)
d1.tconfint_mean(0.05) # 


# ## 6.2 假设检验与单样本T检验
# 当年住宅价格的增长率是否超过了10%的阈值

# In[9]:

d1 = sm.stats.DescrStatsW(house_price_gr.rate)
print('t-statistic=%6.4f, p-value=%6.4f, df=%s' %d1.ttest_mean(0.1))
#一般认为FICO高于690的客户信誉较高，请检验该产品的客户整体信用是否高于690


# ## 6.3 两样本T检验
# - 数据集描述与属性说明
# - ID	客户编号
# - Suc_flag	成功入网标识
# - ARPU	入网后ARPU
# - PromCnt12	12个月内的营销次数
# - PromCnt36	36个月内的营销次数
# - PromCntMsg12	12个月内发短信的次数
# - PromCntMsg36	36个月内发短信的次数
# - Class	客户重要性等级(根据前运营商消费情况)
# - Age	年龄
# - Gender	性别
# - HomeOwner	是否拥有住房
# - AvgARPU	当地平均ARPU
# - AvgHomeValue	当地房屋均价
# - AvgIncome	当地人均收入

# 导入数据

# In[10]:

camp= pd.read_csv(r'tele_camp_ok.csv', skipinitialspace=True)
camp.head()


# 检验当地客户平均客户价值对是否入网的影响

# In[11]:

camp['AvgARPU'].groupby(camp['Suc_flag']).describe()


# - 第一步:方差齐次检验

# In[12]:

Suc0 = camp[camp['Suc_flag'] == 0]['AvgARPU']
Suc1 = camp[camp['Suc_flag'] == 1]['AvgARPU']
leveneTestRes = stats.levene(Suc0, Suc1, center='median')
print('w-value=%6.4f, p-value=%6.4f' %leveneTestRes)


# - 第二步:T-test

# In[13]:

stats.stats.ttest_ind(Suc0, Suc1, equal_var=False)
# Or Try: sm.stats.ttest_ind(gender0, gender1, usevar='pooled')

#测试一下营销次数对是否响应的作用.
#camp['PromCnt12'].groupby(camp['Suc_flag']).describe()
#Suc0 = camp[camp['Suc_flag'] == 0]['PromCnt12']
#Suc1 = camp[camp['Suc_flag'] == 1]['PromCnt12']
#leveneTestRes = stats.levene(Suc0, Suc1, center='median')
#print('w-value=%6.4f, p-value=%6.4f' %leveneTestRes)
#stats.stats.ttest_ind(Suc0, Suc1, equal_var=False)

# ## 6.4 方差分析
# - 单因素方差分析

# In[14]:

pd.set_option('display.max_columns', None) # 设置显示所有列
camp.groupby('Class')[['ARPU']].describe().T


# In[15]:

# 利用回归模型中的方差分析
import statsmodels.api as sm
from statsmodels.formula.api import ols

sm.stats.anova_lm(ols('ARPU ~ C(Class)',data=camp).fit())


# - 多因素方差分析

# In[16]:不考虑交互相

sm.stats.anova_lm(ols('ARPU ~ C(Class)+C(Gender)',data=camp).fit())
# In[16]:考虑交互相
sm.stats.anova_lm(ols('ARPU ~ C(Class)+C(Gender)+C(Class)*C(Gender)',data=camp).fit())


# ## 6.5 相关分析
# 散点图

# In[ ]:

camp.plot(x='AvgARPU', y='ARPU', kind='scatter')

# In[ ]:
import numpy as np
camp['AvgARPU_ln']=np.log(camp['AvgARPU'])
camp['ARPU_ln']=np.log(camp['ARPU'])
#%%
camp.plot(x='AvgARPU_ln', y='ARPU_ln', kind='scatter')
# 相关性分析:“spearman”,“pearson” 和 "kendall"

# In[ ]:
camp[['AvgARPU_ln', 'ARPU_ln']].corr(method='pearson')

# ## 6.6卡方检验

# In[7]:

cross_table = pd.crosstab(camp.Class, columns=camp.Suc_flag)
# Or try this: accepts.pivot_table(index='bankruptcy_ind',columns='bad_ind', values='application_id', aggfunc='count')
cross_table


# In[9]:

cross_table_rowpct = cross_table.div(cross_table.sum(1),axis = 0)
cross_table_rowpct


# In[ ]:

print('chisq = %6.4f\n p-value = %6.4f\n dof = %i\n expected_freq = %s'  %stats.chi2_contingency(cross_table))


# In[ ]:




