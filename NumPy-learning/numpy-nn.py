import numpy as np

# 本例来自 PyTorch 官方网站

N = 64
D_in = 1000
H = 100
D_out = 10

# Return a sample (or samples) from the "standard normal" distribution.
x = np.random.randn(N, D_in)  # 64 * 1000
y = np.random.randn(N, D_out)  # 64 * 10

# 权重矩阵

W1 = np.random.randn(D_in, H)  # 1000 * 100
W2 = np.random.randn(H, D_out)  # 100 * 10

learning_rate = 1e-6

for t in range(500):
    # 前向传播，计算 y 的预测值
    h = x.dot(W1)  # 64*100
    h_relu = np.maximum(h, 0)  # 64*100
    y_pred = h_relu.dot(W2)  # 64*10

    # 计算损失
    loss = np.square(y_pred - y).sum()
    print("迭代次数：", t, "损失：", loss)

    # 梯度反向传播
    grad_y_pred = 2 * (y_pred - y)  # 64*10
    grad_w2 = h_relu.T.dot(grad_y_pred)  # 100*64 * 64*10
    grad_h_relu = grad_y_pred.dot(W2.T)  # 64*10 * 10*100
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    W1 -= learning_rate * grad_w1
    W2 -= learning_rate * grad_w2
