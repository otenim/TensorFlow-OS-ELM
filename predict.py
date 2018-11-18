import argparse
import os
import tqdm
import time
import numpy as np
from os_elm import OS_ELM

parser = argparse.ArgumentParser()
parser.add_argument('--n_input_nodes', type=int, default=784)
parser.add_argument('--n_hidden_nodes', type=int, default=32)
parser.add_argument('--n_output_nodes', type=int, default=784)
parser.add_argument('--loss', default='mean_absolute_error')
parser.add_argument('--activation', default='linear')
parser.add_argument('--n', type=int, default=10000)

def main(args):

    os_elm = OS_ELM(
        n_input_nodes=args.n_input_nodes,
        n_hidden_nodes=args.n_hidden_nodes,
        n_output_nodes=args.n_output_nodes,
        loss=args.loss,
        activation=args.activation,
    )

    for batch_size in [1, 4, 8, 16, 32, 64, 128, 256]:
        pbar = tqdm.tqdm(total=args.n, desc='Batchsize %d' % batch_size)
        times = []
        for i in range(args.n):
            x = np.random.normal(size=(batch_size, args.n_input_nodes))
            t = np.random.normal(size=(batch_size, args.n_output_nodes))
            stime = time.time()
            os_elm.evaluate(x, t)
            etime = time.time()
            times.append(etime - stime)
            pbar.update(1)
        pbar.close()
        times.sort()
        times = times[:args.n // 10]
        times = np.array(times)
        mean = np.mean(times)
        print('mean prediction time: %f [msec/batch]' % (1000*mean))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
