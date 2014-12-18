__author__ = 'bingo4508'
# -*- coding: UTF-8 -*-
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import numpy as np
import math


# 0 = 鼠, 1 = 牛, 2 = 虎, 3 = 兔, 4 = 龍, 5 = 蛇, 6 = 馬, 7 = 羊, 8 = 猴, 9 = 雞, 10 = 狗, 11 = 豬
# 12 = 一, 13 = 二, 14 = 三, 15 = 四, 16 = 五, 17 = 六, 18 = 七, 19 = 八, 20 = 九, 21 = 十
# 22 = 壹, 23 = 貳, 24 = 參, 25 = 肆, 26 = 伍, 27 = 陸, 28 = 柒, 29 = 捌, 30 = 玖, 31 = 拾

def error_track_0(predict_label, answer_label):
    err = 0
    for m, n in zip(predict_label, answer_label):
        if m != n:
            if m <= 11:
                err += 1
            elif m+10 != n and m-10 != n:   # Not same number
                err += 1
    return float(err)/len(answer_label)


def error_track_1(predict_label, answer_label):
    pass


def load_data(fn):
    return load_svmlight_file(fn)


# array is a list of 1X12810 (ex) numpy sparse matrix
# Usage e.g:
# X_train, y_train = load_svmlight_file("../dataset/original/ml14fall_train.dat")
# draw(X_train[0:10])
def draw(array, width=105, flat=True):
    if flat is True:
        fig = []
        for a in array:
            li = a.todense().ravel().tolist()[0]
            nrows = len(li)/width
            r = []
            for i in range(nrows):
                r.append(np.array([li[i*width:i*width+width]]))
            fig.append(np.concatenate([e for e in r]))
    else:
        fig = [e for e in array] if type(array) is list else [array]
    nrows = math.ceil(math.sqrt(len(fig)))
    for i, e in enumerate(fig):
        plt.subplot(nrows, nrows, i)
        plt.axis('off')
        plt.imshow(e, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


# Transform a one-dimensional sparse array to a n x m dense matrix
def to_matrix(array, width=105):
    fig = []
    for a in array:
        li = a.todense().ravel().tolist()[0]
        nrows = len(li)/width
        r = []
        for i in range(nrows):
            r.append(np.array([li[i*width:i*width+width]]))
        fig.append(np.concatenate([e for e in r]))
    return fig if len(fig) > 1 else fig[0]


# x is a n x m np matrix (narray)
def remove_blank(x):
    brow = [i for i, row in enumerate(x) if sum(row) == 0]
    bcol = [i for i, row in enumerate(x.T) if sum(row) == 0]
    y = np.delete(x, brow, 0)
    y = np.delete(y, bcol, 1)
    return y


# Change every element in narray to 1.0
def to_black(x):
    return np.array([[math.ceil(e) for e in r] for r in x])


# x is a n x m np matrix (narray)
def to_square_shape(x):
    h, w = x.shape[0], x.shape[1]
    k = abs((h - w)/2)
    if h > w:
        y = np.concatenate((np.array([[0]*h]*k), x.T))
        y = np.concatenate((y, np.array([[0]*h]*k)))
        y = y.T
    else:
        y = np.concatenate((np.array([[0]*w]*k), x))
        y = np.concatenate((y, np.array([[0]*w]*k)))
    return y


