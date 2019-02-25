import matplotlib.pyplot as plt
import numpy as np
# 在图中显示函数的标签（label）
# show_function_label

x = np.linspace(0, 2 * np.pi, 100)

# plt.plot(x, np.sin(x), 'r-')
# plt.plot(x, np.cos(x), 'b-')
# 多次的 plot 会叠加
# 上面的两行可以写在一行中

# plt.plot(x, np.sin(x), 'r-', x, np.cos(x), 'b-')
# plt.plot(x, np.sin(x), 'r-', label='$sin$')
# plt.plot(x, np.cos(x), 'b-', label='$cos$')
# 如果要显示标签，一定要加上 plt.legend()
# plt.legend()


plt.plot(x, np.sin(x), 'r-', x, x + 1, 'b-')
# 下面这样写也可以，但是可读性不强
plt.legend(['$y = sin(x)$', '$y = x + 1$'])
plt.show()
