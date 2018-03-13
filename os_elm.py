import numpy as np
import tqdm
import tensorflow as tf

class OS_ELM(object):

    def __init__(
        self, n_input_nodes, n_hidden_nodes, n_output_nodes,
        activation='sigmoid', loss='mean_squared_error', name=None):
        if name == None:
            self.name = 'model'
        else:
            self.name = name
        self.__sess = tf.Session()
        self.__is_finished_init_train = False
        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes
        self.__n_output_nodes = n_output_nodes
        if activation == 'sigmoid':
            self.__activation = tf.nn.sigmoid
        elif activation == 'linear':
            self.__activation = tf.identity
        else:
            raise ValueError(
                'an unknown activation function \'%s\' was given. '
                'we currently support \'sigmoid\' and \'linear\'.' % (activation)
            )
        if loss == 'mean_squared_error':
            self.__loss = tf.losses.mean_squared_error
        elif loss == 'mean_absolute_error':
            self.__loss = tf.keras.losses.mean_absolute_error
        elif loss == 'binary_crossentropy':
            self.__loss = tf.keras.losses.binary_crossentropy
        elif loss == 'categorical_crossentropy':
            self.__loss = tf.keras.losses.categorical_crossentropy
        else:
            raise ValueError(
                'an unknown loss function \'%s\' was given. '
                'we currently support \'mean_squared_error\', \'mean_absolute_error\', '
                '\'binary_crossentropy\', \'categorical_crossentropy\'.' % loss
            )
        self.__x = tf.placeholder(tf.float32, shape=(None, self.__n_input_nodes), name='x')
        self.__t = tf.placeholder(tf.float32, shape=(None, self.__n_output_nodes), name='t')
        self.__alpha = tf.constant(
            np.random.uniform(-1., 1., size=(self.__n_input_nodes, self.__n_hidden_nodes)),
            dtype=tf.float32,
            name='alpha',
        )
        self.__bias = tf.constant(
            np.random.uniform(-1., 1., size=(self.__n_hidden_nodes)),
            dtype=tf.float32,
            name='bias',
        )
        self.__p = tf.Variable(
            initial_value=np.zeros(shape=(self.__n_hidden_nodes, self.__n_hidden_nodes)),
            trainable=False,
            dtype=tf.float32,
            name='p',
        )
        self.__beta = tf.Variable(
            initial_value=np.zeros(shape=(self.__n_hidden_nodes, self.__n_output_nodes)),
            trainable=False,
            dtype=tf.float32,
            name='beta',
        )

        # Predict
        self.__predict = tf.matmul(self.__activation(tf.matmul(self.__x, self.__alpha) + self.__bias), self.__beta)

        # Loss
        self.__loss = self.__loss(self.__t, self.__predict)

        # Accuracy
        self.__accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.__predict, 1), tf.argmax(self.__t, 1)), tf.float32))

        # Initial training phase
        self.__init_train = self.__build_init_train_graph()

        # Sequential training phase
        self.__seq_train = self.__build_seq_train_graph()

        # Initialize variables
        self.__sess.run(tf.global_variables_initializer())

    def predict(self, x):
        return self.__sess.run(self.__predict, feed_dict={self.__x: x})

    def evaluate(self, x, t, metrics=['loss']):
        ret = []
        for metric in metrics:
            if metric == 'loss':
                loss =  self.__sess.run(self.__loss, feed_dict={self.__x: x, self.__t: t})
                ret.append(loss)
            elif metric == 'accuracy':
                accuracy = self.__sess.run(self.__accuracy, feed_dict={self.__x: x, self.__t: t})
                ret.append(accuracy)
            else:
                continue
        return ret


    def init_train(self, x, t):
        if self.__is_finished_init_train:
            raise Exception(
                'the initial training phase has already finished. '
                'please call \'seq_train\' method for further training.'
            )
        if len(x) < self.__n_hidden_nodes:
            raise ValueError(
                'in the initial training phase, the number of training samples '
                'must be greater than the number of hidden nodes. '
                'But this time len(x) = %d, while n_hidden_nodes = %d' % (len(x), self.__n_hidden_nodes)
            )
        self.__sess.run(self.__init_train, feed_dict={self.__x: x, self.__t: t})
        self.__is_finished_init_train = True

    def seq_train(self, x, t):
        if self.__is_finished_init_train == False:
            raise Exception(
                'you have not gone through the initial training phase yet. '
                'please first initialize the model\'s weights by \'init_train\' '
                'method before calling \'seq_train\' method.'
            )
        self.__sess.run(self.__seq_train, feed_dict={self.__x: x, self.__t: t})

    def __build_init_train_graph(self):
        H = self.__activation(tf.matmul(self.__x, self.__alpha) + self.__bias)
        HT = tf.transpose(H)
        HTH = tf.matmul(HT, H)
        p = tf.assign(self.__p, tf.matrix_inverse(HTH))
        pHT = tf.matmul(p, HT)
        pHTt = tf.matmul(pHT, self.__t)
        init_train = tf.assign(self.__beta, pHTt)
        return init_train

    def __build_seq_train_graph(self):
        H = self.__activation(tf.matmul(self.__x, self.__alpha) + self.__bias)
        HT = tf.transpose(H)
        HTH = tf.matmul(HT, H)
        batch_size = tf.shape(self.__x)[0]
        I = tf.eye(batch_size)
        Hp = tf.matmul(H, self.__p)
        HpHT = tf.matmul(Hp, HT)
        temp = tf.matrix_inverse(I + HpHT)
        pHT = tf.matmul(self.__p, HT)
        p = tf.assign(self.__p, self.__p - tf.matmul(tf.matmul(pHT, temp), Hp))
        pHT = tf.matmul(p, HT)
        Hbeta = tf.matmul(H, self.__beta)
        seq_train = self.__beta.assign(self.__beta + tf.matmul(pHT, self.__t - Hbeta))
        return seq_train

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
