__author__ = 'bingo4508'

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from util import *
from datetime import datetime
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

TRAIN_DATA = 'train_p2_50'
MODEL_NAME = 'train_p2_50_pca'


TEST = True
TEST_SIZE = 0.5
PCA = True


print "Loading data..."
X_train, y_train = load_data("../dataset/%s.dat" % TRAIN_DATA)

# Split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=TEST_SIZE, random_state=0)

# Preprocess --------------------------------------------------------------
if PCA:
    print "PCA..."
    t1 = datetime.now()

    pca = RandomizedPCA(n_components=100)
    X_train = pca.fit_transform(X_train.todense())
    X_test = pca.transform(X_test.todense())

    print "PCAing %f secs" % (datetime.now() - t1).total_seconds()
else:
    X_train = X_train.todense()
    X_test = X_test.todense()

# Train--------------------------------------------------------------
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 1e-2, 1e-3], 'C': [10, 100, 1000]}]
print "Training..."
t1 = datetime.now()

if TEST:
    # clf = GridSearchCV(SVC(), tuned_parameters, cv=5, verbose=2).fit(X_train, y_train)
    # print clf.best_estimator_

    clf = RandomForestClassifier(n_estimators=500, oob_score=True)
    clf.fit(X_train, y_train)
else:
    clf = SVC(kernel='rbf', C=10, gamma=0.001)
    clf.fit(X_train, y_train)

joblib.dump(clf, '../model/svm/%s.pkl' % MODEL_NAME)

print "Training %f secs" % (datetime.now() - t1).total_seconds()

tlabel = clf.predict(X_test)
if TEST:
    print 'Error: %f' % error_track_0(tlabel, y_test)