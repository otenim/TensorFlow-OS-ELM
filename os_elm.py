import numpy as np
import pickle as pkl
import tqdm

class OS_ELM(object):

    def __init__(
        self, n_input_nodes, n_hidden_nodes, n_output_nodes,
        activation='sigmoid', loss='mean_squared_error',
        alpha_init=None, beta_init=None, bias_init=None):

        # =====================================
        # Model architecture
        # =====================================
        # number of nodes for each layer
        self.n_input_nodes = n_input_nodes
        self.n_hidden_nodes = n_hidden_nodes
        self.n_output_nodes = n_output_nodes
        self.p = None
        self.__is_finished_init_phase = False

        # loss function
        self.__loss_name = loss
        self.__loss = self.__get_loss_function(self.__loss_name)

        # activation function
        self.__activation_name = activation
        self.__activation = self.__get_activation_function(self.__activation_name)

        # =====================================
        # Initialize Weights
        # =====================================
        # check initial weight matrix for alpha
        if alpha_init:
            if alpha_init.shape != (self.n_input_nodes, self.n_hidden_nodes):
                raise ValueError(
                    'alpha_init.shape must be (n_input_nodes, n_hidden_nodes) '
                    '= (%d, %d), but this time alpha_init.shape = %s' \
                        % (self.n_input_nodes, self.n_hidden_nodes, str(alpha_init.shape))
                )
            self.alpha = alpha_init
        else:
            # uniform distribution (min=-1.0, max=1.0)
            self.alpha = np.random.rand(
                self.n_input_nodes,
                self.n_hidden_nodes
            ) * 2.0 - 1.0

        # check initial weight matrix for beta
        if beta_init:
            if beta_init.shape != (self.n_input_nodes, self.n_hidden_nodes):
                raise ValueError(
                    'beta_init.shape must be (n_hidden_nodes, n_output_nodes) '
                    '= (%d, %d), but this time beta_init.shape = %s' \
                        % (self.n_hidden_nodes, self.n_output_nodes, str(beta_init.shape))
                )
            self.beta = beta_init
        else:
            # uniform distribution (min=-1.0, max=1.0)
            self.beta = np.random.rand(
                n_hidden_nodes,
                n_output_nodes
            ) * 2.0 - 1.0

        # check initial weight vector for bias
        if bias_init:
            if bias_init.shape != (self.n_hidden_nodes,):
                raise ValueError(
                    'bias_init.shape must be (n_hidden_nodes,) = '
                    '(%d,), but this time bias_init.shape = %s' \
                        % (self.n_hidden_nodes, str(bias_init.shape))
                )
            self.bias = bias_init
        else:
            # initialize with zeros
            self.bias = np.zeros(shape=(self.n_hidden_nodes))

    def __call__(self, x):
        h = x.dot(self.alpha) + self.bias
        h = self.__activation(h)
        y = h.dot(self.beta)
        return y

    def init_train(self, x, y):
        if len(x) <= self.n_hidden_nodes:
            raise ValueError(
                'the number of samples for initial training should be greater '
                'than the number of hidden nodes (i.e., len(x) > n_hidden_nodes), '
                'but this time len(x) = %d while n_hidden_nodes = %d.' % (len(x), self.n_hidden_nodes)
            )
        H = self.__activation(x.dot(self.alpha) + self.bias)
        HT = H.T
        self.p = np.linalg.pinv(HT.dot(H))
        self.beta = self.p.dot(HT).dot(y)
        self.__is_finished_init_phase = True

    def seq_train(self, x, y, batch_size=16, verbose=1):
        if self.__is_finished_init_phase == False:
            raise Exception(
                'please execute \'init_train\' before \'seq_train\'.'
            )

        n = len(x)
        if verbose:
            pbar = tqdm.tqdm(total=n, desc='sequential training')
        for i in range(0, n, batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            H = self.__activation(x_batch.dot(self.alpha) + self.bias)
            HT = H.T
            I = np.eye(len(x_batch))

            # update P
            temp = np.linalg.pinv(I + H.dot(self.p).dot(HT))    # temp.shape = (N, N)
            self.p = self.p - (self.p.dot(HT).dot(temp).dot(H).dot(self.p))

            # update beta
            self.beta = self.beta + (self.p.dot(HT).dot(y_batch - H.dot(self.beta)))
            if verbose:
                pbar.update(n=len(x_batch))

        if verbose:
            pbar.close()

    def get_model_params(self):
        params = {
            'n_input_nodes': self.n_input_nodes,
            'n_hidden_nodes': self.n_hidden_nodes,
            'n_output_nodes': self.n_output_nodes,
            'loss': self.__loss_name,
            'activation': self.__activation_name,
            'alpha': self.alpha,
            'beta': self.beta,
            'p': self.p,
            'bias': self.bias
        }
        return params

    def save(self, filepath):
        model_params = self.get_model_params()
        with open(filepath, 'wb') as f:
            pkl.dump(model_params, f)

    def predict(self, x, softmax=False):
        if softmax:
            return self.__softmax(self(x))
        else:
            return self(x)

    def evaluate(self, x, y, metrics=['loss']):
        y_true = y
        y_pred = self(x)
        ret = []
        for m in metrics:
            if m == 'accuracy':
                acc = self.__accuracy(y_true, y_pred)
                ret.append(acc)
            elif m == 'loss':
                loss = self.__loss(y_true, y_pred)
                ret.append(loss)
        return ret

    def __accuracy(self, y_true, y_pred):
        batch_size = len(y_pred)
        acc = np.sum((np.argmax(y_pred, axis=-1) == np.argmax(y_true, axis=-1)))
        return acc / batch_size

    def __mean_squared_error(self, y_true, y_pred):
        return 0.5 * np.mean((y_true - y_pred)**2)

    def __mean_absolute_error(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def __linear(self, x):
        return x

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __softmax(self, x):
        c = np.max(x, axis=1).reshape(-1, 1)
        upper = np.exp(x - c)
        lower = np.sum(upper, axis=1).reshape(-1, 1)
        return upper / lower

    def __get_activation_function(self, name):
        if name == 'sigmoid':
            return self.__sigmoid
        elif name == 'linear':
            return self.__linear
        else:
            raise ValueError('unknown activation function \'%s\' was given' % name)

    def __get_loss_function(self, name):
        if name == 'mean_squared_error':
            return self.__mean_squared_error
        elif name == 'mean_absolute_error':
            return self.__mean_absolute_error
        else:
            raise ValueError('unknown loss function \'%s\' was given.' % name)

def load_model(filepath):
    with open(filepath, 'r') as f:
        params = pkl.load(f)
        os_elm = OS_ELM(
            n_input_nodes=params['n_input_nodes'],
            n_hidden_nodes=params['n_hidden_nodes'],
            n_output_nodes=params['n_output_nodes'],
            loss=params['loss'],
            activation=params['activation'],
            alpha_init=params['alpha'],
            beta_init=params['beta'],
            bias_init=params['bias'],
            p_init=params['p']
        )
        return
