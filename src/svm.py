__author__ = 'bingo4508'

from sklearn import cross_validation
from sklearn.svm import SVC
from util import *
import time


TRAIN_SIZE = 9000
X_train, y_train = load_data("../dataset/original/ml14fall_train.dat")

X_train_s, y_train_s = X_train[:TRAIN_SIZE], y_train[:TRAIN_SIZE]
X_test_s, y_test_s = X_train[TRAIN_SIZE:], y_train[TRAIN_SIZE:]
# cross_validation.train_test_split(X_train_s, y_train_s, test_size=0.4, random_state=0)

# Train
t0 = time.clock()
clf = SVC(gamma=0.001)
clf.fit(X_train_s, y_train_s)
print "Training %f secs" % (time.time() - t0)

# Test
t0 = time.clock()
tlabel = clf.predict(X_test_s)
err_rate = error_track_0(tlabel, y_test_s)
print "Testing %f secs" % (time.time() - t0)

print err_rate