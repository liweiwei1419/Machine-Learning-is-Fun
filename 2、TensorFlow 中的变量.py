import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')

f = x * x * y + y + 2

# 运行

sess = tf.Session()
# 变量运行之前一定要初始化
sess.run(x.initializer)
sess.run(y.initializer)

result = sess.run(f)
print('在 session 中执行变量的计算', result)
# 最后一定要记得关闭 session
sess.close()
