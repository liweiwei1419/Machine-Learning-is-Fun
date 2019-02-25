#!/bin/python
'''
Date: 20160406
@author: zhaozhiyong
'''


def f(x):
    x_1 = x[0]
    x_2 = x[1]
    x_3 = x[2]

    result = 2 * (x_1 * x_1) + (x_2 * x_2) + (x_3 * x_3) + 3

    return result


if __name__ == "__main__":
    # print "hello world"
    err = 1.0e-10
    x = [1.0, 1.0, 1.0]
    f_0 = f(x)
    while 1:
        # print "Hello"
        x[0] = x[1] + x[2]
        x[1] = x[0] / 2 - x[2]
        x[2] = x[0] / 3 - 2 * x[1] / 3

        f_t = f(x)

        if (abs(f_t - f_0) < err):
            break

        f_0 = f_t

    print("max: " + str(f_0))
    print(x)
