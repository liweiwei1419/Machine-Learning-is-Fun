from matplotlib import pyplot as plt
import numpy as np

X = np.linspace(-np.pi, np.pi, 200, endpoint=True)
C, S = np.cos(X), np.sin(X)

ax = plt.gca()
# 移动坐标轴边线
# 坐标轴总共有四个连线，我们通过设置透明色隐藏上方和右方的边线
# 通过 set_position() 移动左侧和下侧的边线
# 通过 set_ticks_position() 设置坐标轴的刻度线的显示位置
ax = plt.gca()  # gca 代表当前坐标轴，即 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
# 设置坐标刻度的字体大小，增加半透明背景
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(16)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65))

# 设置坐标轴的长度
plt.xlim(X.min() * 1.1, X.max() * 1.1)
plt.ylim(C.min() * 1.1, C.max() * 1.1)

# 设置坐标轴的刻度和标签
plt.xticks((-np.pi, -np.pi / 2, np.pi / 2, np.pi),
           label=(r'$-\pi$', r'$-\pi/2$', r'$+\pi/2$', r'$+\pi$'))
plt.yticks([-1, -0.5, 0, 0.5, 1])


# 画出余弦曲线，并设置线条颜色，宽度，样式
plt.plot(X, C, color="blue", linewidth=2.0, linestyle="-",label='$y=cos(x)$')
# 画出正弦曲线，并设置线条颜色，宽度，样式
plt.plot(X, S, color="red", linewidth=2.0, linestyle="-",label='$y=sin(x)$')



# 在坐标轴上标示相应的点
t = 2 * np.pi / 3
# 画出 cos(t) 所在的点在 X 轴上的位置，即画出 (t, 0) -> (t, cos(t)) 线段，使用虚线
plt.plot([t, t], [0, np.cos(t)], color='blue', linewidth=1.5, linestyle="--")
# 画出标示的坐标点，即在 (t, cos(t)) 处画一个大小为 50 的蓝色点
plt.scatter([t, ], [np.cos(t), ], 50, color='blue')
# 画出标示点的值，即 cos(t) 的值
plt.annotate(r'$cos(\frac{2\pi}{3})=-\frac{1}{2}$',
             xy=(t, np.cos(t)), xycoords='data',
             xytext=(-90, -50), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
# 画出 sin(t) 所在的点在 X 轴上的位置，即画出 (t, 0) -> (t, sin(t)) 线段，使用虚线
plt.plot([t, t], [0, np.sin(t)], color='red', linewidth=1.5, linestyle="--")
# 画出标示的坐标点，即在 (t, sin(t)) 处画一个大小为 50 的红色点
plt.scatter([t, ], [np.sin(t), ], 50, color='red')
# 画出标示点的值，即 sin(t) 的值
plt.annotate(r'$sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
             xy=(t, np.sin(t)), xycoords='data',
             xytext=(+10, +30), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))


# 在左上角添加铭牌
plt.legend(loc='upper left')
# 把结果显示在屏幕上
plt.show()