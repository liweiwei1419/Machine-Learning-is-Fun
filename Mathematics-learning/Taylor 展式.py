# Taylor 展式的应用，计算指数函数
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt


# 计算 e^x，其中 x 较小（该方法被调用）
def _calc_e_small(x):
    n = 10
    f = np.arange(1, n + 1).cumprod()  # 累乘
    b = np.array([x] * n).cumprod()
    return np.sum(b / f) + 1


# 计算 e^x，其中 x 任意
def calc_e(x):
    # 默认的情况下，x > 0 的时候，不用求导数
    reverse = False
    if x < 0:
        x = -x
        reverse = True

    ln2 = math.log(2)
    c = x / ln2
    a = int(c + 0.5)  # 这是一个整数
    b = x - a * ln2
    y = (2 ** a) * _calc_e_small(b)
    if reverse:
        return 1 / y
    return y


if __name__ == '__main__':
    t1 = np.linspace(-2, 0, 10, endpoint=False)
    t2 = np.linspace(0, 2, 20)
    print(len(t1))
    print(len(t2))
    t = np.concatenate((t1, t2))
    # print(t)
    print(len(t))

    y = np.empty_like(t)
    for i, x in enumerate(t):
        y[i] = calc_e(x)
        print('e^{} = {}（近似值），{} （真实值）'.format(x, y[i], math.exp(x)))

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(t, y, 'r-', t, y, 'go', linewidth=2)
    plt.title('Taylor 展式的应用', fontsize=18)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('$e^x$', fontsize=15)
    plt.grid(b=True)
    plt.show()
