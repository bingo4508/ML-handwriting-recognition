__author__ = 'bingo4508'
# -*- coding: UTF-8 -*-
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import numpy as np


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


# array is a 1X12810 (ex) numpy sparse matrix
# Usage e.g:
# X_train, y_train = load_svmlight_file("../dataset/original/ml14fall_train.dat")
# draw(X_train[0])
def draw(array, width=105):
    li = array.todense().ravel().tolist()[0]
    nrows = len(li)/width
    r = []
    for i in range(nrows):
        r.append(np.array([li[i*width:i*width+width]]))
    r = np.concatenate([e for e in r])
    plt.imshow(r, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()