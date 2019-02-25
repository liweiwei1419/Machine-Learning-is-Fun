import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from matplotlib.figure import SubplotParams

# 使用多项式回归拟合正弦函数（Pipeline）

sns.set()

n_dots = 200

X = np.linspace(-2 * np.pi, 2 * np.pi, n_dots)
Y = np.sin(X) + 0.2 * np.random.rand(n_dots) - 0.1
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

plt.plot(X, Y, 'go')
plt.show()


def polynomial_features(degree):
    polynomial_features = PolynomialFeatures(degree=degree)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    return pipeline


degrees = [2, 3, 4, 5, 6, 7, 8, 9]

results = []

plt.figure(dpi=100)
for d in degrees:
    model = polynomial_features(degree=d)

    model.fit(X, Y)

    train_score = model.score(X, Y)

    y_pred = model.predict(X)

    mse = mean_squared_error(Y, y_pred)
    results.append({"model": model, "degree": d, "score": train_score, "mse": mse})

for r in results:
    print(r)

# 下面开始绘图

# plt.figure(figsize=(12, 6), dpi=200, subplotpars=SubplotParams(hspace=0.3))

for i, r in enumerate(results):
    fig = plt.subplot(4, 2, i + 1)
    plt.xlim(-8, 8)
    plt.title("LinearRegression degree={}".format(r['degree']))
    plt.scatter(X, Y, s=5, c='b', alpha=0.5)
    plt.plot(X, r["model"].predict(X), 'r--')
plt.show()
