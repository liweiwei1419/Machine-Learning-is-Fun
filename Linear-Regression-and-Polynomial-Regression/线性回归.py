import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 采样点(Xi,Yi)
Xi = np.array([8.19, 2.72, 6.39, 8.71, 4.7, 2.66, 3.78])
Yi = np.array([7.01, 2.78, 6.47, 6.71, 4.1, 4.23, 4.05])

lr = LinearRegression()

lr.fit(Xi.reshape(-1, 1), Yi)

line_X = np.linspace(0, 10, 100)
line_y = lr.predict(line_X.reshape(-1, 1))

plt.scatter(Xi, Yi, color="red", label="Sample Point", linewidth=3)
plt.plot(line_X, line_y, color="orange", label="Fitting Line", linewidth=2)
plt.show()
