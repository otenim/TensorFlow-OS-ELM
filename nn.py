import numpy as np
import tqdm
import tensorflow as tf
import os

class NN(object):
    def __init__(
        self, n_input_nodes, n_hidden_nodes, n_output_nodes,
        hidden_activation='relu', output_activation='sigmoid',
        loss='mean_squared_error', optimizer='adam', name=None):

        if name == None:
            self.name = 'model'
        else:
            self.name = name

        self.__sess = tf.Session()
        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes
        self.__n_output_nodes = n_output_nodes
        self.__hidden_actfun = self.__get_activation(hidden_activation)
        self.__output_actfun = self.__get_activation(output_activation)
        self.__lossfun = self.__get_loss(loss)
        self.__optimizer = self.__get_optimizer(optimizer)
        self.__x = tf.placeholder(tf.float32, shape=(None, self.__n_input_nodes), name='x')
        self.__t = tf.placeholder(tf.float32, shape=(None, self.__n_output_nodes), name='t')
        self.__w1 = tf.get_variable(
            'w1',
            shape=[self.__n_input_nodes, self.__n_hidden_nodes],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer()
        )
        self.__w2 = tf.get_variable(
            'w2',
            shape=[self.__n_hidden_nodes, self.__n_output_nodes],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer()
        )
        self.__b = tf.get_variable(
            'b',
            shape=[self.__n_hidden_nodes],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        # Predict
        h1 = self.__hidden_actfun(tf.matmul(self.__x, self.__w1) + self.__b)
        h2 = self.__output_actfun(tf.matmul(h1, self.__w2))
        self.__predict = h2

        # Loss
        self.__loss = self.__lossfun(self.__t, self.__predict)

        # Accuracy
        self.__accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.__predict, 1), tf.argmax(self.__t, 1)), tf.float32))

        # Train
        self.__fit_step = self.__optimizer.minimize(self.__loss)

        # Saver
        self.__saver = tf.train.Saver()

        # Initialize variables
        self.__sess.run(tf.global_variables_initializer())

    def train_on_batch(self, x, t):
        self.__sess.run(self.__fit_step, feed_dict={
            self.__x: x,
            self.__t: t,
        })

    def predict_on_batch(self, x, t):
        self.__sess.run(self.__predict, feed_dict={
            self.__x: x,
        })

    def test_on_batch(self, x, t, metrics=['loss']):
        met = []
        for m in metrics:
            if m == 'loss':
                met.append(self.__loss)
            elif m == 'accuracy':
                met.append(self.__accuracy)
            else:
                raise ValueError(
                    'an unknown metric \'%s\' was given.' % (m)
                )
        ret = self.__sess.run(met, feed_dict={
            self.__x: x,
            self.__t: t,
        })
        return ret

    def __del__(self):
        self.__sess.close()

    @property
    def n_input_nodes(self):
        return self.__n_input_nodes

    @property
    def n_hidden_nodes(self):
        return self.__n_hidden_nodes

    @property
    def n_output_nodes(self):
        return self.__n_output_nodes

    def save(self, filepath):
        self.__saver.save(self.__sess, filepath)

    def restore(self, filepath):
        self.__saver.restore(self.__sess, filepath)

    def reset_variables(self):
        for var in [self.__w1, self.__w2, self.__b]:
            self.__sess.run(var.initializer)

    def __get_optimizer(self, name):
        if name == 'adam':
            return tf.train.AdamOptimizer()
        elif name == 'sgd':
            return tf.train.GradientDescentOptimizer()
        else:
            raise ValueError(
                'an unknown optimizer \'%s\' was given.' % name
            )

    def __get_loss(self, name):
        if name == 'mean_squared_error':
            return tf.losses.mean_squared_error
        elif name == 'mean_absolute_error':
            return tf.keras.losses.mean_absolute_error
        elif name == 'softmax_crossentropy':
            return tf.losses.softmax_cross_entropy
        else:
            raise ValueError(
                'an unknown loss function \'%s\' was given.' % name
            )

    def __get_activation(self, name):
        if name == 'relu':
            return tf.nn.relu
        elif name == 'sigmoid':
            return tf.nn.sigmoid
        elif name == 'linear':
            return tf.identity
        else:
            raise ValueError(
                'an unknown activation function \'%s\' was given. ' % (name)
            )
