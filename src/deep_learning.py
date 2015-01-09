from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
import cv2

from util import *
from sklearn.externals import joblib
from datetime import datetime


TRAIN_DATA = 'train_jdong_64.dat'
MODEL_NAME = 'train_jdong_64.dat'

TEST_SIZE = 0.5
DUMP = False

##############################################################################
if __name__ == '__main__':
    print "Loading data..."
    X_train, y_train = load_data("../dataset/%s" % TRAIN_DATA)

    # Split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=TEST_SIZE, random_state=0)
    X_train = X_train.todense()
    X_test = X_test.todense()


    # Train --------------------------------------------------------------
    print "Training..."
    t1 = datetime.now()

    dbn = DBN(
        [-1, 300, 300, -1],
        learn_rates=0.1,
        learn_rate_decays=0.9,
        epochs=10,
        verbose=1)
    dbn.fit(X_train, y_train)


    print "Training %f secs" % (datetime.now() - t1).total_seconds()

    if TEST_SIZE > 0:
        tlabel = dbn.predict(X_test)
        print 'Error: %f' % error_track_0(tlabel, y_test)

    if DUMP:
        # Dump model --------------------------------------------------------------
        print "Dumping model..."
        joblib.dump(dbn, '../model/deep/%s.pkl' % MODEL_NAME)
