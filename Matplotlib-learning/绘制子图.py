import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['STHeiti']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

x = np.arange(0.0, 5, 0.01)
y = np.cos(2 * np.pi * x)
plt.plot(x, y, label='$sin(x)$', color='red', linewidth=2)
plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5), arrowprops=dict(facecolor='black', shrink=0.05), )
plt.ylim(-2, 2)

plt.title('正弦波')
plt.xlabel('时间（秒）')
plt.ylabel('电压')
plt.axis([0, 5, -2, 2])
plt.legend()
plt.show()
