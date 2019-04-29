from matplotlib import pyplot as plt
import numpy as np

def plt_scatter():
    n = 1024
    X = np.random.normal(0, 1, n)
    Y = np.random.normal(0, 1, n)
    T = np.arctan2(Y, X)

    plt.subplot(1, 2, 1)
    plt.scatter(X, Y, s=75, c=T, alpha=.5)

    plt.xlim(-1.5, 1.5)
    plt.xticks(())
    plt.ylim(-1.5, 1.5)
    plt.yticks(())

def plt_fill_between():
    n = 256
    X = np.linspace(-np.pi, np.pi, n, endpoint=True)
    Y = np.sin(2 * X)

    plt.subplot(1, 2, 2)

    plt.plot(X, Y + 1, color='blue', alpha=1.00)
    plt.fill_between(X, 1, Y + 1, color='blue', alpha=.25)

    plt.plot(X, Y - 1, color='blue', alpha=1.00)
    plt.fill_between(X, -1, Y - 1, (Y - 1) > -1, color='blue', alpha=.25)
    plt.fill_between(X, -1, Y - 1, (Y - 1) < -1, color='red',  alpha=.25)

    plt.xlim(-np.pi, np.pi)
    plt.xticks(())
    plt.ylim(-2.5, 2.5)
    plt.yticks(())

plt.figure(figsize=(16, 6))
plt_scatter()
plt_fill_between()
plt.tight_layout()
plt.show()
