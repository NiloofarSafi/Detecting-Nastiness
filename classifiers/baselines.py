# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from random import randint

__all__=['BaselineBadWordRatioClassifier']


class BaselineBadWordRatioClassifier(BaseEstimator, ClassifierMixin):
    """
    Baseline classifier
    """
    def __init__(self, T=0.1):
        self.T = T

    def fit(self, X, Y=None):

        self.classes_ = [0, 1]  # 0-> negative 1-> positive
        return self

    def _decision_function(self, X):
        y_predicted = []
        for x in X:
            # y_predicted.append(self.classes_[0])
            # y_predicted.append(self.classes_[randint(0,1)])
            if np.any(x >= self.T):
                y_predicted.append(self.classes_[0])
            else:
                y_predicted.append(self.classes_[1])
            '''if x[2] > 0:
                y_predicted.append(self.classes_[1])
                #if x[2] > 0:
                #    y_predicted.append(self.classes_[1])
                #else:
                #    y_predicted.append(self.classes_[0])
            else:
                if x[4] >= self.T:
                    y_predicted.append(self.classes_[0])
                else:
                    y_predicted.append(self.classes_[1])'''

        return np.array(y_predicted)

    def predict(self, X):
        return self._decision_function(X)