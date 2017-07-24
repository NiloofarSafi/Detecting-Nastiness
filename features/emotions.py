# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


__all__=['EmoticonsFeature']

emoticons = {
    'happy': ["<3", "<333", ":d", ":dd", "8)", ":-)", ":)", ";)", "(-:", "(:", ":o)", ':c)', 'o_o', '>:p', ':-p', ':p',
              'x-p', 'xp', ':-p', '=p', ':-', '❤', '^_^', ],
    'sad': [":/", ":>", ":')", ":-(", ":(", ":S", ":-S", ':-||', ':@', '>:(', '</3']}


class EmoticonsFeature(BaseEstimator,TransformerMixin):
    """
    Emoticon Feature estimator
    """
    def __init__(self, emo=None):
        if not emo:
            self.emoticon_dict = emoticons
        else:
            self.emoticon_dict= emo


    def get_feature_names(self):
        return np.array(['Feature_happy_emo', "Feature_unhappy_emo", "Feature_emo_total"])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        fea_happy_emo, fea_sad_emo, fea_emo = [], [], []
        for doc in documents:
            happy_emo = 0
            sad_emo = 0
            total = 0
            #print(doc.id)
            for emo in doc.emoticons:
                #print(doc.emoticons)
                emo = emo.strip().lower()
                if emo in self.emoticon_dict['happy']:
                    happy_emo += 1
                if emo in self.emoticon_dict['sad']:
                    sad_emo += 1
                total += 1
                print("finished")
            fea_happy_emo.append(happy_emo)
            fea_sad_emo.append(sad_emo)
            fea_emo.append(total)
        #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        X = np.array([fea_happy_emo, fea_sad_emo, fea_emo]).T
        return X