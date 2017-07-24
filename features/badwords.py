# -*- coding: utf-8 -*-
from __future__ import print_function, division
import codecs
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import os
from preprocess import ark_tweet_tokenizer
from preprocess.twokenize import regex_or
import logging
import re

log = logging.getLogger(__name__)

__all__ = ['BadWordRatio']

emoticons = {
    'happy': ["<3", "<333", ":d", ":dd", "8)", ":-)", ":)", ";)", "(-:", "(:", ":o)", ':c)', 'o_o', '>:p', ':-p', ':p',
              'x-p', 'xp', ':-p', '=p', ':-', '^_^', ],
    'sad': [":/", ":>", ":')", ":-(", ":(", ":S", ":-S", ':-||', ':@', '>:(', '</3'],
    'smiley' : ["<3", "<333", ":-)", ":)", "(-:", "(:", ":o)", ':c)']}

class BadWordRatio(BaseEstimator, TransformerMixin):
    """
    Baseline feature estimator
    """
    _bad_words = os.path.join(os.path.dirname(__file__), 'resources')

    def __init__(self, bad_word_file='bad_word.txt', emo=None):
        self._bad_word_file = bad_word_file
        self._load_bad_words(os.path.join(self._bad_words, self._bad_word_file))
        if not emo:
            self.emoticon_dict = emoticons
        else:
            self.emoticon_dict=emo

    def get_feature_names(self):
        return np.array(["Feature_happy_emo", "Feature_unhappy_emo", "Feature_smiley_emo", "Feature_emo_total", "negative_word_ratio"])

    def fit(self, X, Y=None):
        return self

    def _load_bad_words(self, file):
        bad_word_list = set()
        with codecs.open(file, mode='r') as f:
            for line in f:
                bad_word_list.add(line.strip(" \r\n"))
        self.bad_word_list_ = set(bad_word_list)
        self.bad_word_list_=set([word.replace('*','.*').replace('+','.+').replace('(','\(') for word in self.bad_word_list_])
        # print (regex_or(*self.bad_word_list_))

        self.pattern_=re.compile(regex_or(*self.bad_word_list_),re.UNICODE)

    def transform(self, X):
        fea_happy_emo, fea_sad_emo, fea_smiley_emo, fea_emo, bad_word_ratio_lst = [], [], [], [], []
        for x in X:
            happy_emo = 0
            sad_emo = 0
            smiley_emo = 0
            total = 0
            for emo in x.emoticons:
                emo = emo.strip().lower()
                if emo in self.emoticon_dict['happy']:
                    happy_emo += 1
                if emo in self.emoticon_dict['sad']:
                    sad_emo += 1
                if emo in self.emoticon_dict['smiley']:
                    smiley_emo += 1
                total += 1
            fea_happy_emo.append(happy_emo)
            fea_sad_emo.append(sad_emo)
            fea_smiley_emo.append(smiley_emo)
            fea_emo.append(total)
            tokens = ark_tweet_tokenizer(x.content)
            bad_word_count = 0
            for token in tokens:
                # if token.lower() in self.bad_word_list_:
                if self.pattern_.match(token.lower()):
                    bad_word_count += 1
            try:

                negative_word_ratio = round(bad_word_count / len(tokens), 2)

            except ZeroDivisionError as ex:
                log.warn('Division by zero. Probably the post is empty: Post length={post} Setting negative_word_ratio to zero'.format(
                    post=len(tokens)))
                negative_word_ratio = 0

            bad_word_ratio_lst.append(negative_word_ratio)

        X_transformed = np.array([fea_happy_emo, fea_sad_emo, fea_smiley_emo, fea_emo, bad_word_ratio_lst]).T
        return X_transformed
