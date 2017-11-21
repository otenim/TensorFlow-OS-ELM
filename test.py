import os
import argparse
import datasets
import models
import numpy as np

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument(
    'batch_model',
    choices=[
        'mnist_mlp',
        'mnist_cnn',
        'fashion_mlp',
        'fashion_cnn'
    ])
parser.add_argument('batch_weights')
parser.add_argument('speed_weights')
parser.add_argument('speed_units', type=int)
parser.add_argument('--batch_size', type=int, default=32)

def compute_score(out):
    out = np.sort(out, axis=1)
    first = out[:,-1]
    second = out[:,-2]
    score = first - second # (samples, 1)
    return score

def main(args):

    # instantiate model and dataset
    if args.batch_model == 'mnist_mlp':
        model = models.create_mnist_mlp()
        dataset = datasets.get_dataset('mnist')
        data = dataset.load_data()
    elif args.batch_model == 'mnist_cnn':
        model = models.create_mnist_cnn()
        dataset = datasets.get_dataset('mnist')
        data = dataset.load_data(network='cnn')
    elif args.batch_model == 'fashion_mlp':
        model = models.create_fashion_mlp()
        dataset = datasets.get_dataset('mnist')
        data = dataset.load_data()
    elif args.batch_model == 'fashion_cnn':
        model = models.create_fashion_cnn()
        dataset = datasets.get_dataset('mnist')
        data = dataset.load_data(network='cnn')
    else:
        raise Exception('unknown model was specified.')
    model.load_weights(args.batch_weights)

    os_elm = models.OS_ELM(
        inputs=dataset.inputs,
        units=args.speed_units,
        outputs=dataset.outputs,
        activation='sigmoid',
        loss='mean_squared_error')
    os_elm.load_weights(args.speed_weights)

    # prepare dataset
    (_, _), (x_test, y_test) = data
    split = 0.5
    border = int(split * len(x_test))
    x_test_train = x_test[:border]
    x_test_test = x_test[border:]
    y_test_train = y_test[:border]
    y_test_test = y_test[border:]

    # test loop
    print('before: %f' % os_elm.compute_accuracy(x_test_test, y_test_test))
    for i in range(0, len(x_test), args.batch_size):
        x_batch = x_test_train[i:i+args.batch_size]
        y_batch = y_test_train[i:i+args.batch_size]
        out = os_elm(x_batch)
        mask = (np.argmax(out,axis=1) != np.argmax(y_batch,axis=1))
        miss = int(np.sum(mask))
        if miss > 0:
            print('miss %d in test data[%d~%d]' % (miss,i,i+args.batch_size))
            x = x_batch[mask]
            y = model.predict_on_batch(x)
            os_elm.seq_train(x,y)
    print('after: %f' % os_elm.compute_accuracy(x_test_test, y_test_test))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
