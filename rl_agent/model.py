import tensorflow as tf
import tensorflow.contrib.slim as slim


class Model:
    def __init__(self, size_y, size_x, num_actions, batch_size):
        self._size_y = size_y
        self._size_x = size_x
        self._dimensions = (size_y, size_x, 2)
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        # now setup the model
        self._define_model()

    def _define_model(self):
        self._states = tf.placeholder(shape=[None, *self._dimensions], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        self._input = slim.flatten(self._states)
        fc1 = tf.layers.dense(self._input, 64, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 64, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc2, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def predict_one(self, sess, state):
        return sess.run(self._logits, feed_dict={self._states: state.reshape(1, *self._dimensions)})

    def predict_batch(self, sess, states):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, states, q_s_a):
        sess.run(self._optimizer, feed_dict={self._states: states, self._q_s_a: q_s_a})

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def var_init(self):
        return self._var_init
