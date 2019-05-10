# pokemon-challenge

数据集：https://www.kaggle.com/terminus7/pokemon-challenge

---

Kernel：https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners

科赛网：

1、https://www.kesci.com/home/dataset/5a26650136ae5c1293bf241a/project

2、[Data Science入门教程](https://www.kesci.com/home/project/5c1b4b54f8caa6002bc7abe9)。

还可以谷歌搜索“口袋妖怪数据集分析”，使用卷积神经网络。

### 逻辑与

```python
data[np.logical_and(data['Defense']>200, data['Attack']>100 )]
```

```python
data[(data['Defense']>200) & (data['Attack']>100)]
```

### 转换为类别变量

```python
data['Type 1'] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')
```

### 绘图

```python
# subplots
data1 = data[['Attack','Defense','Speed']]
data1.plot(subplots=True)
plt.show()
```

热力图

```python

```

### 一个星号和两个星号

```python
def f(*args):
    for i in args:
        print(i)


f(1)
print("")
f(1, 2, 3, 4)
```

```python
def f(**kwargs):
    for key, value in kwargs.items():
        print(key, " ", value)


f(country='spain', capital='madrid', population=123456)
```

