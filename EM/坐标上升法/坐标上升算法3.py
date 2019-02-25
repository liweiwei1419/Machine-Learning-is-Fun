# 坐标上升算法
# http://blog.csdn.net/lch614730/article/details/17055577
# define
def f(x1, x2, x3):
    return -x1 * x1 - 2 * x2 * x2 - 3 * x3 * x3 + 2 * x1 * x2 + 2 * x1 * x3 - 4 * x2 * x3 + 6


x1 = 1.
x2 = 1.
x3 = 1.
f0 = f(x1, x2, x3)
err = 0.0001
while True:
    x1 = x2 + x3
    x2 = 0.5 * x1 - x3
    x3 = 1.0 / 3 * x1 - 2.0 / 3 * x2
    ft = f(x1, x2, x3)
    if (abs(ft - f0) < err):
        break
    else:
        f0 = ft
print("x1:{}".format(x1))
print("x2:{}".format(x2))
print("x3:{}".format(x3))
print('最值：{}'.format(f(x1, x2, x3)))
