import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim


class QNetwork:
    def __init__(self, field_size, num_actions):
        # None shapes are for batch sizes
        size_final_layer = 128
        self.input = tf.placeholder(shape=[None, field_size, field_size, 1], dtype=tf.float32)
        self.conv1 = slim.conv2d(inputs=self.input,
                                 num_outputs=64,
                                 kernel_size=[3, 3],
                                 stride=[1, 1],
                                 padding='VALID',
                                 biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1,
                                 num_outputs=size_final_layer,
                                 kernel_size=[2, 2],
                                 stride=[1, 1],
                                 padding='VALID',
                                 biases_initializer=None)

        # Take output of convolution and create fully connected layer, if understood correctly
        self.stream = slim.flatten(self.conv2)
        xavier_init = tf.contrib.layers.xavier_initializer()
        # First dimension is batch_size
        self.W = tf.Variable(xavier_init([self.stream.get_shape().as_list()[1], num_actions]))
        self.Qout = tf.matmul(self.stream, self.W)
        self.predict = tf.argmax(self.Qout, 1)

        # Create action layer
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


class ExperienceBuffer:
    """Experience Buffer contains samples with (state, action, reward, next_state, done)"""

    def __init__(self, buffer_size=10000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


def update_target_graph(tfvars, tau):
    total_vars = len(tfvars)
    op_holder = []
    for idx, var in enumerate(tfvars[0:total_vars//2]):
        op_holder.append(tfvars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfvars[idx+total_vars//2].value())))
    return op_holder


def update_target(op_holder, sess):
    for op in op_holder:
        sess.run(op)
