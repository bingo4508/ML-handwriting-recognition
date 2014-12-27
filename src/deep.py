__author__ = 'bingo4508'
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from util import *
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
import time


TRAIN_DATA = 'train_p2_50'
MODEL_NAME = 'train_p2_50'
TEST_SIZE = 0.5
Evaluate = True

X_train, y_train = load_data("../dataset/%s.dat" % TRAIN_DATA)
if Evaluate:
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=TEST_SIZE, random_state=0)

# Models we will use
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

clf = Pipeline([("rbm", rbm), ("logistic", logistic)])

# Training RBM-Logistic Pipeline
params = {
    "rbm__learning_rate": [0.1, 0.01, 0.001],
    "rbm__n_iter": [20, 40, 80],
    "rbm__n_components": [50, 100, 200],
    "logistic__C": [1.0, 10.0, 100.0]}  # perform a grid search over the parameter

start = time.time()
gs = GridSearchCV(clf, params, n_jobs=5,  verbose=1)
gs.fit(X_train, y_train)

print "\ndone in %0.3fs" % (time.time() - start)
print "best score: %0.3f" % (gs.best_score_)
print "RBM + LOGISTIC REGRESSION PARAMETERS"
bestParams = gs.best_estimator_.get_params()

# loop over the parameters and print each of them out
# so they can be manually set
for p in sorted(params.keys()):
    print "\t %s: %f" % (p, bestParams[p])

# Evaluation
if Evaluate is True:
    print("Error:\n%f\n" % error_track_0(gs.predict(X_test), y_test))

    # print("Logistic regression using raw pixel features:\n%s\n" % error_track_0(logistic_clf.predict(X_test), y_test))
else:
    # Save model
    joblib.dump(gs, '../model/deep/%s_RBM.pkl' % MODEL_NAME)
    # joblib.dump(logistic_clf, '../model/deep/%s_raw.pkl' % MODEL_NAME)
    # ##############################################################################