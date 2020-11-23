# -*- coding: utf-8 -*-
"""
Execution dans un terminal

Exemple:
   python3 src/classifier.py 1 0

Bougeard Yann 20137996
Wilmo Maël 20 138 003
"""

import numpy as np
import sys
from parser import parser_train, parser_test, sub_writer
from sklearn.model_selection._split import StratifiedShuffleSplit
from sklearn import preprocessing
from models.SVM import SVM
from models.MLP import MLP
from models.LDA import LDA
from models.logistic import Logistic
from models.ridge import LinearRidge
from models.perceptron import ClassicPerceptron

CV_NUM = 5


def compute_error(Y_pred, Y):
    """Return the percentage of data that wasn't in the good class"""
    err = 0
    for i in range(0, len(Y_pred)):
        if Y_pred[i] != Y[i]:
            err += 1
    return err / len(Y_pred) * 100


def main():

    if len(sys.argv) < 1:
        usage = "\n Usage: python3 src/classifier.py model\
        \n\n\t model: name of the model you want to use\
        \n\t svm, "
        print(usage)
        return

    model_name = sys.argv[1]

    # Get the data from the training set and the testing set
    train_X, train_y = parser_train()
    sub_X, sub_id = parser_test()
    labels = sorted(list(set(train_y)))

    train_X = preprocessing.scale(train_X)
    sub_X = preprocessing.scale(sub_X)

    
    # Get the good model
    model = None
    if model_name == "svm":
        model = SVM
    elif model_name == "mlp":
        model = MLP
    elif model_name == "lda":
        model = LDA
    elif model_name == "logistic":
        model = Logistic
    elif model_name == "ridge":
        model = LinearRidge
    elif model_name == "perceptron":
        model = ClassicPerceptron
    else:
        raise Exception("This model was not implemented")

    # Split the data in CV_NUM sets of training data and validation data
    # with equal proportions of each classes
    # For cross-validation
    print("===== Cross-Validation ====")
    sss = StratifiedShuffleSplit(n_splits=CV_NUM, test_size=0.2)

    ind = 1
    err_train = []
    err_valid = []
    models = []
    for train_index, valid_index in sss.split(train_X, train_y):
        X_train, X_valid = \
            train_X[train_index], train_X[valid_index]
        y_train, y_valid = train_y[train_index], train_y[valid_index]
        
        the_model = model()
        the_model.fit(X_train, y_train)
        train_pred = the_model.predict(X_train)
        test_pred = the_model.predict(X_valid)

        err_train.append(compute_error(train_pred, y_train))
        err_valid.append(compute_error(test_pred, y_valid))

        print("  CV ", ind, "/", CV_NUM)
        # print("    X_train samples: ", len(X_train))
        print("    Training   error: ", err_train[-1], "%")
        print("    Validation error: ", err_valid[-1], "%")
        ind += 1
        models.append(the_model)

    print("")
    print("==== Mean ====")
    print("  Training   error: ", np.mean(err_train), "%")
    print("  Validation error: ", np.mean(err_valid), "%")

    # Using the mean of the output of all the models to improve prediction
    # preds_sum = models[0].predict(sub_X)

    # for i in range(1, len(models)):
    #     preds_sum = np.sum([preds_sum, models[i].predict(sub_X)], axis=0)

    # preds = [x / CV_NUM for x in preds_sum]

    # Write the results in the submission file
    # sub_writer(preds, sub_id, labels)


if __name__ == "__main__":
    main()
