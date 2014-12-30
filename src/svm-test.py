__author__ = 'bingo4508'

from util import *
from datetime import datetime
from sklearn.externals import joblib
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler


TRAIN_DATA = 'train_p5_from_p2_normalize_100'
TEST_DATA = 'test_p5_from_p2_normalize_100'
MODEL_NAME = 'train_p5_from_p2_normalize_100'
PCA = False
SCALE = False

###############################################################################
if __name__ == '__main__':
    # Load data
    print "Loading data..."
    X_test, y_test = load_data("../dataset/%s.dat" % TEST_DATA)
    if PCA or SCALE:
        X_train, y_train = load_data("../dataset/%s.dat" % TRAIN_DATA)
    else:
        X_test = X_test.todense()

    # Load model
    print "Loading model..."
    clf = joblib.load('../model/svm/%s.pkl' % MODEL_NAME)

    if SCALE:
        scaler = StandardScaler()
        scaler.fit_transform(X_train)
        scaler.transform(X_test)
    if PCA:
        print "PCA and Scale..."
        pca = RandomizedPCA(n_components=100)
        X_train = pca.fit_transform(X_train.todense())
        X_test = pca.transform(X_test.todense())


    # Test
    print "Testing..."
    t1 = datetime.now()

    tlabel = clf.predict(X_test)
    print 'Error: %f' % error_track_0(tlabel, y_test)

    print "Testing %f secs" % (datetime.now() - t1).total_seconds()

    # Output result
    with open('../output/%s_result.dat' % TEST_DATA, 'w') as f:
        for e in tlabel:
            f.write("%d\n" % e)
