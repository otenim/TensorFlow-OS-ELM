import numpy as np
import os
import pickle as pkl

def min_max_normalize(x, axis=None):
    min = np.min(x, axis=axis)
    max = np.max(x, axis=axis)
    return (x-min)/(max-min)

def z_normalize(x, axis=None):
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    return (x - mean)/std

def save_result(result, result_dir):
    fname = '%s_units%d_bsize%d.txt' % (
        result['dataset'],
        result['units'],
        result['batch_size'])
    with open(os.path.join(result_dir, fname), 'w') as f:
        f.write('********** Results **********\n')
        f.write('dataset: %s\n' % result['dataset'])
        f.write('epochs: %d\n' % (result['epochs']))
        f.write('batch_size: %d\n' % (result['batch_size']))
        f.write('units: %d\n' % (result['units']))
        f.write('activation: %s\n' % (result['activation']))
        f.write('mean_init_train_time: %.5f[sec]\n' % (result['mean_init_train_time']))
        f.write('mean_seq_train_time: %.5f[sec/batch]\n' % (result['mean_seq_train_time']))
        f.write('mean_pred_time: %.5f[sec/batch]\n' % (result['mean_pred_time']))
        f.write('mean_test_loss: %.5f\n' % (result['mean_test_loss']))
        if result.get('mean_test_acc'):
            f.write('mean_test_acc: %.5f\n' % (result['mean_test_acc']))
        f.write('********** Record **********\n')
        f.write('|init [sec]|seq [sec/batch]|pred [sec/batch]|loss|*acc|\n')
        f.write('<tr>\n')
        f.write('\t<td>%.5f</td>\n' % (result['mean_init_train_time']))
        f.write('\t<td>%.5f</td>\n' % (result['mean_seq_train_time']))
        f.write('\t<td>%.5f</td>\n' % (result['mean_pred_time']))
        f.write('\t<td>%.5f</td>\n' % (result['mean_test_loss']))
        if result.get('mean_test_acc'):
            f.write('\t<td>%.5f</td>\n' % (result['mean_test_acc']))
        f.write('</tr>\n')
