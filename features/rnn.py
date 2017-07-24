from __future__ import division, print_function
from nltk_contrib.readability.readabilitytests import ReadabilityTool
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


__all__ = ['RNNFeatures']


class RNNFeatures(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        #return np.array(['rnn_vec'])
        return np.array(['rnn_' + str(i) for i in range(self.num_features_)])

    def fit(self, documents, y=None):
        self.num_features_ = documents[0].rnn_vector.shape[0]
        print ("The shape of rnn vector is : {0}".format( documents[0].rnn_vector.shape))
        print(self.num_features_)
        return self

    def make_feature_vec(self, doc):

        # Function to average all of the word vectors in a given
        # paragraph
        #
        # Pre-initialize an empty numpy array (for speed)
        #print("hereeeeeeeeeeeeeeeeeee")
        feature_vec = np.zeros((self.num_features_,), dtype=np.float32)
        #
        counter = 0

        for item in doc.rnn_vector:
            feature_vec[counter] = item
            counter += 1

        return feature_vec

    def transform(self, documents):
        doc_feature_vecs = np.zeros((len(documents), self.num_features_), np.float32)
        count=0
        for i, doc in enumerate(documents):

            if not np.count_nonzero(doc.rnn_vector)>0:
                count+=1
            doc_feature_vecs[i] = self.make_feature_vec(doc)
        # rnn_vec = []
        # for doc in documents:
        #     rnn_vec.append(doc.rnn_vector)
        #     # print(doc.rnn_vector.shape)
        # X = np.array(rnn_vec)
        print ("The shape of feature matrix is: {0}".format(doc_feature_vecs.shape))
        print ("The zero vectors are : {0}".format(count))
        return doc_feature_vecs
