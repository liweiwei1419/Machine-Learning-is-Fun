import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')

f = x * x * y + y + 2

# f.eval() 这样也可以

with tf.Session() as sess:
    # 相当于调用 tf.get_default_session().run(x.initial)
    x.initializer.run()
    y.initializer.run()
    # result = sess.run(f)
    # f.eval() 相当于调用 tf.get_default_session().run(f)
    result = f.eval()
    print(result)
