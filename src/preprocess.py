__author__ = 'bingo4508'

from util import *
import skimage.transform
from skimage.morphology import skeletonize

WIDTH = 50


def preprocess_1(sparse_matrix):
    l = None
    for i, e in enumerate(sparse_matrix):
        print "preprocess_1: %d" % i
        x = skimage.transform.resize(to_square_shape(remove_blank(to_matrix(e))), (WIDTH, WIDTH))
        x.resize((1, WIDTH**2))
        l = x if l is None else np.concatenate((l, x))
    return l


def preprocess_2(sparse_matrix):
    l = None
    for i, e in enumerate(sparse_matrix):
        print "preprocess_2: %d" % i
        x = to_black(skimage.transform.resize(to_square_shape(remove_blank(to_matrix(e))), (WIDTH, WIDTH)))
        x.resize((1, WIDTH**2))
        l = x if l is None else np.concatenate((l, x))
    return l


def preprocess_3(sparse_matrix):
    l = None
    for i, e in enumerate(sparse_matrix):
        print "preprocess_3: %d" % i
        x = 1*np.resize(skeletonize(to_matrix(e, width=WIDTH)), (1, WIDTH**2))
        l = x if l is None else np.concatenate((l, x))
    return l


def remove_noise(m):
    pass


if __name__ == '__main__':
    data, label = load_data("../dataset/original/ml14fall_test1_no_answer.dat")
    # data, label = load_data("../dataset/original/ml14fall_train.dat")
    # data, label = load_data("../dataset/train_p2.dat")

    tdata = preprocess_2(data)
    dump_data(tdata, label, "../dataset/test_p2_50.dat")

    tdata = preprocess_1(data)
    dump_data(tdata, label, "../dataset/test_p1_50.dat")