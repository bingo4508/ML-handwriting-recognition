__author__ = 'bingo4508'

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

from util import *
from datetime import datetime
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

TRAIN_DATA = 'train_jdong_HoG_64'
MODEL_NAME = 'train_jdong_HoG_64'

TEST_SIZE = 0
DUMP = True

PCA = False
SCALE = False

##############################################################################
if __name__ == '__main__':
    print "Loading data..."
    X_train, y_train = load_data("../dataset/%s.scale" % TRAIN_DATA)

    # Split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=TEST_SIZE, random_state=0)
    X_train = X_train.todense()
    X_test = X_test.todense()

    # Preprocess --------------------------------------------------------------
    if SCALE:
        print "Scaling..."
        scaler = StandardScaler()
        scaler.fit_transform(X_train)
        scaler.transform(X_test)

    if PCA:
        print "PCA..."
        t1 = datetime.now()

        pca = RandomizedPCA(n_components=100)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        print "PCAing %f secs" % (datetime.now() - t1).total_seconds()


    # Train --------------------------------------------------------------
    print "Training..."
    t1 = datetime.now()
    clf = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=1, random_state=0)
    scores = cross_val_score(clf, X_train, y_train)
    print "cross val: ", scores
    clf.fit(X_train, y_train)
    print "Training %f secs" % (datetime.now() - t1).total_seconds()

    if TEST_SIZE > 0:
        tlabel = clf.predict(X_test)
        print 'Error: %f' % error_track_0(tlabel, y_test)

    if DUMP:
        # Dump model --------------------------------------------------------------
        print "Dumping model..."
        joblib.dump(clf, '../model/rf/%s.pkl' % MODEL_NAME)
