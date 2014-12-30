__author__ = 'bingo4508'

from util import *
import skimage.transform
from skimage.morphology import skeletonize
from skimage.feature import hog
from multiprocessing import Process
from scipy.sparse import vstack
from skimage.filter.rank import median
from skimage.morphology import disk
from skimage.filter import gaussian_filter
import os
import time


################### Preprocess methods ##############################

def square_resize(m, width, flat=True):
    if flat:
        m = to_matrix(m, width=width)
    return skimage.transform.resize(to_square_shape(remove_blank(m)), (width, width))


def normalize(m, width, flat=True):
    if flat:
        m = to_matrix(m, width=width)
    return skimage.transform.resize(remove_blank(m), (width, width))


def square_resize_all_black(m, width, flat=True):
    if flat:
        m = to_matrix(m, width=width)
    return to_black(skimage.transform.resize(to_square_shape(remove_blank(m)), (width, width)))


def normalize_all_black(m, width, flat=True):
    if flat:
        m = to_matrix(m, width=width)
    return to_black(skimage.transform.resize(remove_blank(m), (width, width)))


def skeleton(m, width, flat=True):
    if flat:
        m = to_matrix(m, width=width)
    return 1*np.resize(skeletonize(m), (1, width**2))


def diagonal(m, width, flat=True):
    zone_size = 10
    if flat:
        m = to_matrix(m, width=width)
    # Get diagonal feature of (width/zone_size)^2 zones
    diag_avg = []
    for i in range(width/zone_size):
        for j in range(width/zone_size):
            # tm: zone_size x zone_size matrix
            tm = m[i*zone_size:i*zone_size + zone_size, j*zone_size:j*zone_size + zone_size]
            s = 0
            for k in range(zone_size):
                s += sum([tm[k-kk, kk] for kk in range(k+1)])
            for k in range(zone_size-1, 0, -1):
                s += sum([tm[zone_size-1-kk, k+kk] for kk in range(zone_size-k)])
            diag_avg.append(float(s)/(2*zone_size-1))
    return np.array([diag_avg])


# HoG
def HoG(m, width, flat=True):
    if flat:
        m = to_matrix(m, width=width)
    return np.array([hog(m, orientations=8, pixels_per_cell=(20, 20), cells_per_block=(1, 1))])


def jdong(m, width, flat=True):
    if flat:
        m = to_matrix(m, width=105)
    # Normalization
    m = square_resize_all_black(m, WIDTH, False)
    m = median(m, disk(2))
    m = normalize(m, WIDTH, False)
    m = median(m, disk(2))
    m = gaussian_filter(m, sigma=0.5)
    m = skimage.transform.resize(m, (64, 64))
    return np.resize(m, (1, 64**2))
    # Feature extraction
    #...


##########################################################################################

def preprocess(func, sparse_matrix, multiproc=-1, label=None):
    l = None
    for i, e in enumerate(sparse_matrix):
        print i
        x = func(e, WIDTH)
        l = x if l is None else np.concatenate((l, x))
    if multiproc >= 0:
        dump_data(l, label, "%stmp_%d" % (OUTPUT_DIR, multiproc))
    else:
        return l


WIDTH = 64
MULTITASK = 10
OUTPUT_DIR = '../dataset/'
OUTPUT = "../dataset/train_jdong_HoG.dat"
func = HoG

if __name__ == '__main__':
    print "Loading data..."

    # data, label = load_data("../dataset/original/ml14fall_test1_no_answer.dat")
    # data, label = load_data("../dataset/original/ml14fall_train.dat")

    data, label = load_data("../dataset/train_jdong_128.dat")
    # data, label = load_data("../dataset/test_p2.dat")

    ##############################################################################################
    # Multi-process...
    procs = []
    interval = data.shape[0]/MULTITASK
    r = [i*interval for i in range(MULTITASK)]+[data.shape[0]]
    for i in range(len(r)-1):
        p = Process(target=preprocess, args=(func, data[r[i]:r[i+1]], i, label[r[i]:r[i+1]]))    # <= Modify here
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # Merge...
    print "Merging..."
    data = []
    label = []
    for i in range(len(r)-1):
        fn = "%stmp_%d" % (OUTPUT_DIR, i)
        tdata, tlabel = load_data(fn)
        data.append(tdata)
        label.append(tlabel)
        os.remove(fn)
    data = vstack(data)
    label = np.concatenate(label)

    # Dump data...
    print "Dumping..."
    dump_data(data, label, OUTPUT)