import tensorflow as tf
import numpy as np


def module(x):
    print('module usage.')
    return x**2


class Module(object):
    def __init__(self, order=2):
        print('class Module usage')
        self.order = order

    def __call__(self, inputs):
        return inputs**self.order


class Model(object):
    def __init__(self, order=2):
        self.x = tf.placeholder(tf.float32, shape=[None, 1])
        self.order = order

        # module init
        self.module1 = Module(order=self.order)
        self.module2 = Module(order=self.order+1)

    def __call__(self):
        # build model
        return self.module1(inputs=self.x), \
               self.module2(inputs=self.module1(inputs=self.x))


if __name__ == '__main__':
    print('self-test Hello World in ', __file__)

    model = Model(order=1)
    # feed and inference
    x = np.array([[10]])
    feed_dict = {model.x: x}

    sess = tf.Session()
    result = sess.run([model()],
                      feed_dict=feed_dict)
    print(result)