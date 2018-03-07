import numpy as np
import tqdm
import tensorflow as tf

class OS_ELM(object):

    def __init__(self, n_input_nodes, n_hidden_nodes, n_output_nodes):
        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes
        self.__n_output_nodes = n_output_nodes
        self.__activation = tf.nn.sigmoid
        self.__loss = tf.losses.mean_squared_error
        self.__x = tf.placeholder(tf.float32, [None, self.__n_input_nodes])
        self.__y = tf.placeholder(tf.float32, [None, self.__n_output_nodes])
        self.__alpha = tf.constant(
            tf.random_uniform(
                [self.__n_input_nodes, self.__n_hidden_nodes],
                minval=-1.0,
                maxval=1.0,
            )
        )
        self.__beta = tf.Variable(
            tf.random_uniform(
                [self.__n_hidden_nodes, self.__n_output_nodes],
                minval=-1.0,
                maxval=1.0,
            ),
            trainable=False,
        )
        self.__p = tf.Variable(
            tf.zeros([self.__n_hidden_nodes, self.__n_hidden_nodes]),
            trainable=False,
        )
        self.__bias = tf.constant(tf.zeros([self.__n_hidden_nodes]))

        # graph for initial training phase
        self.__init_train_p, self.__init_train_beta = self.__build_init_train_graph()
        # graph for sequential training phase
        self.__seq_train_p, self.__seq_train_beta = self.__build_seq_train_graph()


    def __build_init_train_graph(self):
        H = self.__activation(tf.matmul(self.__x, self.__alpha) + self.__bias)
        HT = tf.transpose(H)

        # update p
        HTH = tf.matmul(H, HT)
        init_train_p = self.__p.assign(tf.matrix_inverse(HTH))

        # update beta
        pHT = tf.matmul(self.__p, HT)
        pHTy = tf.matmul(pHT, self.__y)
        init_train_beta = self.__beta.assign(pHTy)
        return (init_train_p, init_train_beta)

    def __build_seq_train_graph(self):
        H = self.__activation(tf.matmul(self.__x, self.__alpha) + self.__bias)
        HT = tf.transpose(H)
        batch_size = tf.shape(self.__x)[0]
        I = tf.eye(batch_size)

        # update p
        Hp = tf.matmul(H, self.__p)
        HpHT = tf.matmul(Hp, HT)
        IHpHT = tf.matrix_inverse(I + HpHT)
        pHT = tf.matmul(self.__p, HT)
        pHTIHpHT = tf.matmul(pHT, IHpHT)
        pHTIHpHTH = tf.matmul(pHTIHpHT, H)
        pHTIHpHTHp = tf.matmul(pHTIHpHTH, self.__p)
        seq_train_p = self.__p.assign(self.__p - pHTIHpHTHp)

        # update beta
        pHT = tf.matmul(self.__p, HT)
        Hbeta = tf.matmul(H, self.__beta)
        seq_train_beta = self.__beta.assign(tf.matmul(pHT, self.__y - Hbeta))
        return (seq_train_p, seq_train_beta)

    @property
    def n_input_nodes(self):
        return self.__n_input_nodes

    @property
    def n_hidden_nodes(self):
        return self.__n_hidden_nodes

    @property
    def n_output_nodes(self):
        return self.__n_output_nodes
