# 设置 inline 方式，直接把图片画在网页上

# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt

# 在 [0, 2*PI] 之间取 100 个点
x = np.linspace(0, 2 * np.pi, num=100)
# 计算这 100 个点的正弦值，并保存在变量 y
y = np.sin(x)
# 画出 x, y 即是我们的正弦曲线
plt.plot(x, y)
plt.show()