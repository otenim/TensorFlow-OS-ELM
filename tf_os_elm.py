import numpy as np
import tqdm
import tensorflow as tf

class OS_ELM(object):

    def __init__(self, n_input_nodes, n_hidden_nodes, n_output_nodes, activation='sigmoid', loss='mean_squared_error'):
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
            np.random.uniform(-1,1,size=(self.__n_input_nodes, self.__n_hidden_nodes)),
            dtype=tf.float32,
            name='alpha',
        )
        self.__bias = tf.constant(
            np.zeros(shape=(self.__n_hidden_nodes)),
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
        self.__predict = self.__activation(tf.matmul(self.__x, self.__alpha) + self.__bias)

        # Initial training phase
        self.__init_train_p, self.__init_train_beta = self.__build_init_train_graph()

        # Sequential training phase
        self.__seq_train_p, self.__seq_train_beta = self.__build_seq_train_graph()

        # Initialize variables
        self.__sess.run(tf.global_variables_initializer())

    def predict(self, x):
        return self.__sess.run(self.__predict, feed_dict={self.__x: x})

    def fit(self, x, t, batch_size=16, verbose=1):

        if verbose:
            pbar = tqdm.tqdm(total=len(x))

        if self.__is_finished_init_train == False:
            if verbose:
                pbar.set_description('initial training phase')
            if len(x) < self.__n_hidden_nodes:
                raise ValueError(
                    'If this is the first time for you to train the model, '
                    'the number of training samples '
                    'must be greater than the number of hidden nodes.'
                )
            x_init = x[:self.__n_hidden_nodes]
            t_init = t[:self.__n_hidden_nodes]
            self.__sess.run(self.__init_train_p, feed_dict={self.__x: x_init})
            self.__sess.run(self.__init_train_beta, feed_dict={self.__x: x_init, self.__t: t_init})
            self.__is_finished_init_train = True
            if verbose:
                pbar.update(n=len(x_init))
            x_seq = x[self.__n_hidden_nodes:]
            t_seq = t[self.__n_hidden_nodes:]
        else:
            x_seq = x
            t_seq = t

        if verbose:
            pbar.set_description('sequential training phase')
        for i in range(0, len(x_seq), batch_size):
            x_batch = x_seq[i:i+batch_size]
            t_batch = t_seq[i:i+batch_size]
            self.__sess.run(self.__seq_train_p, feed_dict={self.__x: x_batch})
            self.__sess.run(self.__seq_train_beta, feed_dict={self.__x: x_batch, self.__t: t_batch})
            if verbose:
                pbar.update(n=len(x_batch))

        if verbose:
            pbar.close()

    def __build_init_train_graph(self):
        H = self.__activation(tf.matmul(self.__x, self.__alpha) + self.__bias)
        HT = tf.transpose(H)
        HTH = tf.matmul(HT, H)
        init_train_p = self.__p.assign(tf.matrix_inverse(HTH))
        pHT = tf.matmul(self.__p, HT)
        pHTt = tf.matmul(pHT, self.__t)
        init_train_beta = self.__beta.assign(pHTt)
        return (init_train_p, init_train_beta)

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
        seq_train_p = self.__p.assign(self.__p - tf.matmul(tf.matmul(pHT, temp), Hp))
        pHT = tf.matmul(self.__p, HT)
        Hbeta = tf.matmul(H, self.__beta)
        seq_train_beta = self.__beta.assign(self.__beta + tf.matmul(pHT, self.__t - Hbeta))
        return (seq_train_p, seq_train_beta)

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
