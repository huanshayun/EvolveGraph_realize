import pickle
from math import ceil

import numpy as np
from multiprocessing import Pool
import time

import random
import operator
from functools import reduce


def write_pickle(obj, outfile, protocol=-1):
    with open(outfile, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)


def read_pickle(infile):
    with open(infile, 'rb') as f:
        return pickle.load(f)


def read_eval(infile):
    with open(infile) as f:
        return eval(f.read())


def write_str(obj, outfile):
    with open(outfile, 'w') as f:
        f.write(str(obj))


def split_ratio(seq, ratio):
    n = len(seq)
    idx = np.arange(n)
    idx_0 = np.random.choice(idx, int(n * ratio), replace=False)
    idx_1 = np.delete(idx, idx_0)
    return seq[idx_0], seq[idx_1]


def split(ls, ratio):
    if not isinstance(ls, set):
        ls = set(ls)
    head = random.sample(ls, int(len(ls) * ratio))
    tail = list(ls - set(head))
    return head, tail

def split_list(input, batch_size, shuffle_input=True):
    num = ceil(len(input) / batch_size)
    if shuffle_input:
        random.shuffle(input)
    return [input[i * batch_size: (i + 1) * batch_size]
            for i in range(num)]


def get_value(query, key, default=None):
    value = query.get(key)
    return default if value is None else value


def pool_map(f, args, init=None, multiple=False, jobs=4):
    t = time.time()
    pool = Pool(jobs, init)
    if multiple:
        result = pool.starmap(f, args)
    else:
        result = pool.map(f, args)
    print('time {:.2f}'.format(time.time() - t))
    pool.close()
    pool.join()
    return result


def func_iterable(f, iterable):
    return [f(i) for i in iterable]


def list_index(ls, i):
    try:
        return ls.index(i)
    except:
        return None


def prod(ls):
    return reduce(operator.mul, ls, 1)


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item

