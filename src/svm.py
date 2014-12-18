__author__ = 'bingo4508'

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
import skimage.transform
from util import *
import time
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    train_data, train_label = load_data("../dataset/original/ml14fall_train.dat")

    # Split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.4, random_state=0)

    # Preprocess
    print "Preprocessing..."
    t0 = time.clock()

    X_train = [skimage.transform.resize(to_square_shape(remove_blank(to_matrix(e))), (100, 100)) for e in X_train]
    X_test = [skimage.transform.resize(to_square_shape(remove_blank(to_matrix(e))), (100, 100)) for e in X_test]

    print "Preprocessing %f secs" % (time.time() - t0)


    # Train
    print "Training..."
    t0 = time.clock()

    clf = SVC(gamma=0.001)
    clf.fit(X_train, y_train)

    print "Training %f secs" % (time.time() - t0)

    # Test
    print "Testing..."

    t0 = time.clock()
    tlabel = clf.predict(X_test)
    err_rate = error_track_0(tlabel, y_test)
    print "Testing %f secs" % (time.time() - t0)

    print err_rate