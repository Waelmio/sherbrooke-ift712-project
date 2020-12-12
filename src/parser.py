"""
Parser

Bougeard Yann 20137996
Wilmo Maël 20 138 003
"""

import os.path
import numpy as np


def parser_train():
    """read the data/train.csv file
    and send back the train data and targets in two list
    data and target"""
    f = open(os.path.join("data", "train.csv"))
    next(f)
    data = []
    target = []
    for line in f:
        temp = line.replace("\n", "").split(",")
        tmp = []
        target.append(temp[1])
        for x in temp[2:]:
            tmp.append(float(x))
        data.append(tmp)
    f.close()
    return np.array(data), np.array(target)


def parser_test():
    """Read the data/test.csv file
    and send back the test data and id in two lists data, id"""
    f = open(os.path.join("data", "test.csv"))
    next(f)
    data = []
    id = []
    for line in f:
        temp = line.replace("\n", "").split(",")
        tmp = []
        id.append(temp[0])
        for x in temp[1:]:
            tmp.append(float(x))
        data.append(tmp)
    f.close()
    return np.array(data), np.array(id)


def sub_writer(preds, ids, labels):
    f = open("out/submission.csv", "w")
    f.write("id" + ",".join(labels) + "\n")

    for i in range(0, len(preds)):
        f.write(ids[i] + "," + ",".join(labels) + "\n")
    f.close()
