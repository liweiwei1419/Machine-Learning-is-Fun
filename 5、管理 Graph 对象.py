import tensorflow as tf

g2 = tf.Graph()


with g2.as_default():
    x2 = tf.Variable(2)

# 变量之中有一个图对象
print(x2.graph is g2) # True

print(x2.graph is tf.get_default_graph()) # v
