# -*- coding: utf-8 -*-
from __future__ import division, print_function

from manage import app
from sklearn.base import BaseEstimator, TransformerMixin
import os
import joblib
import numpy as np
import scipy.sparse as sp
from manage import app

__all__ = ['FeatureLoader','DumpedFeaturesTransformers']



class FeatureLoader(object):
    """Loads the dumped features objects, vectors and ids

        ids and vectors are parallel array
        ids contains the unique names/id of the book
        vectors contains the corresponding features vector for the book

        Follow lazy loading scheme



    """
    def __init__(self, feature, config):
        self.load_dir =config.VECTORS
        self._feature = feature

    def model(self):
        if not hasattr(self, 'model_'):
            self.model_ = joblib.load(os.path.join(self.load_dir, self._feature + '.model'))
        return self.model_

    def vectors(self):
        if not hasattr(self, 'vectors_'):
            print ("Loading vectors .....")
            self.vectors_ =  joblib.load(os.path.join(self.load_dir, self._feature + '.vector'))
        return self.vectors_


    def ids(self):
        if not hasattr(self, 'ids_'):
            self.ids_ = joblib.load(os.path.join(self.load_dir, self._feature + '.ids'))
        return self.ids_



class DumpedFeaturesTransformers(BaseEstimator, TransformerMixin):
    """
    Loads the dumped features

    """
    # __dumped_dir = current_app.config['VECTORS']
    __dumped_dir = app.VECTORS

    def __init__(self, feature):
        self.feature = feature

        if os.path.exists(os.path.join(self.__dumped_dir, self.feature + '.vector')):
            feature_loader=FeatureLoader(self.feature,app)
            self._X_ids = feature_loader.ids()
            self._vectors = feature_loader.vectors()
            self._model = feature_loader.model()

        else:
            raise ValueError("Feature dump for  %s does not exist in %s" % (
                feature, os.path.join(self.__dumped_dir, feature + '.vector')))

    def get_feature_names(self):
        return self._model.get_feature_names()

    def fit(self, X, y=None):
        return self

    def transform(self, books):
        X = []
        sparse = sp.issparse(self._vectors)
        for book in books:

            if book.book_id in self._X_ids:
                book_index = self._X_ids.index(book.book_id)
                if sparse:
                    X.append(self._vectors[book_index].toarray()[0])
                else:
                    X.append(self._vectors[book_index])
            else:
                # this should not happen
                print("Herer inside danger zone")
                X.append(self._model.transform(book)[0])

        if sparse:
            # print X[0]

            X = sp.csr_matrix(X)
        else:
            X = np.array(X)
        # print X[0]
        return X
