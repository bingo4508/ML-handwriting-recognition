__author__ = 'bingo4508'

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

from util import *
from datetime import datetime
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

TRAIN_DATA = 'train_jdong_HoG'
MODEL_NAME = 'train_jdong_HoG'

VALIDATION = True
TEST_SIZE = 0.5
DUMP = False

PCA = False
SCALE = False

##############################################################################
if __name__ == '__main__':
    print "Loading data..."
    X_train, y_train = load_data("../dataset/%s.dat" % TRAIN_DATA)

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
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 1e-2, 1e-3], 'C': [10, 100, 1000]}]
    print "Training..."
    t1 = datetime.now()
    if VALIDATION:
        clf = GridSearchCV(SVC(), tuned_parameters, cv=5, verbose=2, n_jobs=-1).fit(X_train, y_train)
        print clf.best_estimator_
    else:
        clf = SVC(kernel='rbf', C=10, gamma=0.1)
        clf.fit(X_train, y_train)
    print "Training %f secs" % (datetime.now() - t1).total_seconds()

    if TEST_SIZE > 0:
        tlabel = clf.predict(X_test)
        print 'Error: %f' % error_track_0(tlabel, y_test)

    if DUMP:
        # Dump model --------------------------------------------------------------
        print "Dumping model..."
        joblib.dump(clf, '../model/svm/%s.pkl' % MODEL_NAME)
