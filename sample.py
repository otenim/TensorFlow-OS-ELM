from models import OS_ELM
from datasets import Mnist

INPUT_NODES=784 # Number of input nodes
HIDDEN_NODES=1024 # Number of hidden nodes
OUTPUT_NODES=10 # Number of output nodes
BATCH_SIZE=128 # Batch size NOTE: not have to be fixed-length

os_elm = OS_ELM(
    inputs=INPUT_NODES,
    units=HIDDEN_NODES,
    outputs=OUTPUT_NODES,
    activation='sigmoid', # 'sigmoid' or 'relu'
    loss='mean_squared_error' # we support 'mean_squared_error' only
)

# x_train: shape=(60000,784), dtype=float32, normalized in [0,1]
# y_train: shape=(60000,10), dtype=float32, one-hot-vector format
# x_test: shape=(10000,784), dtype=float32, normalized in [0,1]
# y_test: shape=(10000,10), dtype=float32, one-hot-vector format
dataset = Mnist()
(x_train, y_train), (x_test, y_test) = dataset.load_data()

# Separate training dataset for initial training phase
# and sequential training phase.
# NOTE: batch size of initial training dataset must be
# much greater than the number of hidden units of os_elm.

border = int(HIDDEN_NODES * 1.1)
x_train_init, x_train_seq = x_train[:border], x_train[border:]
y_train_init, y_train_seq = y_train[:border], y_train[border:]

# Initial training phase
print('now initial training phase...')
os_elm.init_train(x_train_init, y_train_init)

# Sequential training phase
print('now sequential training phase...')
for i in range(0, len(x_train_seq), BATCH_SIZE):
    x = x_train_seq[i:i+BATCH_SIZE]
    y = y_train_seq[i:i+BATCH_SIZE]
    os_elm.seq_train(x,y)

# 'forward' method just forward the input data and return the outputs
# This method can be used for any type of problems.
# out.shape = (10000,10)
out = os_elm.forward(x_test)

# 'classify' method forward the input data and return the probability
# for each class.
# NOTE: This method can only be used for classification problems.
# out.shape = (10000,10)
prob = os_elm.classify(x_test)

# Compute loss
# 'compute_loss' method can be used for any problems.
loss = os_elm.compute_loss(x_test,y_test)

# Compute accuracy
# NOTE: 'compute_accuracy' method can only be used for classification problems
acc = os_elm.compute_accuracy(x_test,y_test)

print('test_loss: %f' % (loss))
print('test_accuracy: %f' % (acc))
