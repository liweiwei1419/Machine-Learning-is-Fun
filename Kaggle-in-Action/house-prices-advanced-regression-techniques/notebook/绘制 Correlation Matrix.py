import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

corr_mat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr_mat, vmax=.8, square=True)
plt.show()
