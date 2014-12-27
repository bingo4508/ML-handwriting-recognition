__author__ = 'bingo4508'

from util import *
import skimage.transform
from skimage.morphology import skeletonize


# Focus Square and resize
def preprocess_1(sparse_matrix):
    l = None
    for i, e in enumerate(sparse_matrix):
        print "preprocess_1: %d" % i
        x = skimage.transform.resize(to_square_shape(remove_blank(to_matrix(e))), (WIDTH, WIDTH))
        x.resize((1, WIDTH**2))
        l = x if l is None else np.concatenate((l, x))
    return l


# Focus Square and resize and to all black
def preprocess_2(sparse_matrix):
    l = None
    for i, e in enumerate(sparse_matrix):
        print "preprocess_2: %d" % i
        x = to_black(skimage.transform.resize(to_square_shape(remove_blank(to_matrix(e))), (WIDTH, WIDTH)))
        x.resize((1, WIDTH**2))
        l = x if l is None else np.concatenate((l, x))
    return l


# Skeletonize
def preprocess_3(sparse_matrix):
    l = None
    for i, e in enumerate(sparse_matrix):
        print "preprocess_3: %d" % i
        x = 1*np.resize(skeletonize(to_matrix(e, width=WIDTH)), (1, WIDTH**2))
        l = x if l is None else np.concatenate((l, x))
    return l


# Diagonal feature [with skeleton]
def preprocess_4(sparse_matrix, zone_size=10, skeleton=False):
    l = None
    for ii, e in enumerate(sparse_matrix):
        print "preprocess_4: %d" % ii
        m = to_matrix(e, width=WIDTH)
        if skeleton:
            m = skeletonize(m)

        # Get diagonal feature of (WIDTH/zone_size)^2 zones
        diag_avg = []
        for i in range(WIDTH/zone_size):
            for j in range(WIDTH/zone_size):
                # tm: zone_size x zone_size matrix
                tm = m[i*zone_size:i*zone_size + zone_size, j*zone_size:j*zone_size + zone_size]
                s = 0
                for k in range(zone_size):
                    s += sum([tm[k-kk, kk] for kk in range(k+1)])
                for k in range(zone_size-1, 0, -1):
                    s += sum([tm[zone_size-1-kk, k+kk] for kk in range(zone_size-k)])
                diag_avg.append(float(s)/(2*zone_size-1))
        x = np.array([diag_avg])
        l = x if l is None else np.concatenate((l, x))
    return l


# Intersection
def preprocess_5(sparse_matrix, zone_size=10, skeleton=False):
    pass


def remove_noise(m):
    pass


WIDTH = 100
if __name__ == '__main__':
    # data, label = load_data("../dataset/original/ml14fall_test1_no_answer.dat")
    # data, label = load_data("../dataset/original/ml14fall_train.dat")

    # data, label = load_data("../dataset/train_p2.dat")
    data, label = load_data("../dataset/test_p2.dat")

    tdata = preprocess_4(data, 10, True)
    dump_data(tdata, label, "../dataset/test_p4_skeleton_from_p2.dat")
