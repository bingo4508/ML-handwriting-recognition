__author__ = 'bingo4508'

from util import *
from datetime import datetime
from sklearn.externals import joblib
from sklearn.decomposition import RandomizedPCA


TRAIN_DATA = 'train_p2_50'
TEST_DATA = 'test_p2_50'
MODEL_NAME = 'train_p2_50_pca'
PCA = True

if __name__ == '__main__':
    # Load data
    print "Loading data..."
    X_train, y_train = load_data("../dataset/%s.dat" % TRAIN_DATA)
    X_test, y_test = load_data("../dataset/%s.dat" % TEST_DATA)

    # Load model
    print "Loading model..."
    clf = joblib.load('../model/svm/%s.pkl' % MODEL_NAME)

    # PCA and Scale
    if PCA:
        print "PCA and Scale..."
        pca = RandomizedPCA(n_components=100)
        X_train = pca.fit_transform(X_train.todense())
        X_test = pca.transform(X_test.todense())

    # Test
    print "Testing..."
    t1 = datetime.now()

    tlabel = clf.predict(X_test)

    print "Testing %f secs" % (datetime.now() - t1).total_seconds()

    # Output result
    with open('../output/%s_result.dat' % TEST_DATA, 'w') as f:
        for e in tlabel:
            f.write("%d\n" % e)
