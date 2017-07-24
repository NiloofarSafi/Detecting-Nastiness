# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.base import BaseEstimator, TransformerMixin
import codecs
from preprocess import ark_tweet_tokenizer
from preprocess.twokenize import regex_or
import logging
import numpy as np
import os
import re



__all__=['QuestionAnswerFeature']


log= logging.getLogger(__name__)





class QuestionAnswerFeature(BaseEstimator,TransformerMixin):
    """
    Domain feature estimator
    """
    _bad_words = os.path.join(os.path.dirname(__file__), 'resources')
    def __init__(self):
        self._load_bad_words(os.path.join(self._bad_words,'bad_word.txt'))

    def _load_bad_words(self, file):
        bad_word_list = set()
        with codecs.open(file, mode='r') as f:
            for line in f:
                bad_word_list.add(line.strip(" \r\n"))
        self.bad_word_list_ = list(set(bad_word_list))
        self.bad_word_list_=list(set([word.replace('*','.*').replace('+','.+').replace('(','\(') for word in self.bad_word_list_]))
        # print (regex_or(*self.bad_word_list_))

        self.pattern_=re.compile(regex_or(*self.bad_word_list_),re.UNICODE)

    def get_feature_names(self):
        features=['Question', 'Answer', 'Anonymous', 'User_Mention', 'QP', 'Bad_Word'] +self.bad_word_list_
        return np.array(features)

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        question, answer, anonymous, user, qp, bad_word,bad_words = [], [], [], [], [], [],[]
        dic = {}
        for doc in documents:
            if doc.post_type == 'Question':
                question.append(1)
            else:
                question.append(0)

            if doc.post_type == 'Answer':
                answer.append(1)
            else:
                answer.append(0)
            found_anon = False
            anon = ['anon', 'anonymous']
            for word in anon:
                if word in doc.content.lower():
                    found_anon = True
                    anonymous.append(1)
                    break
            if not found_anon:
                anonymous.append(0)

            if doc.has_proper_noun():
                user.append(1)
            else:
                user.append(0)

            tokens = ark_tweet_tokenizer(doc.content)
            bad_word_count = 0
            bad_word_list=[0]*len(self.bad_word_list_)
            for token in tokens:
                if self.pattern_.match(token.lower()):
                    bad_word_count += 1
                if token.lower() in self.bad_word_list_:
                #     bad_word_count += 1
                    bad_word_list[self.bad_word_list_.index(token.lower())]+=1


            try:

                negative_word_ratio = round(bad_word_count / len(tokens), 2)

            except ZeroDivisionError as ex:
                log.warn('Division by zero. Probably the post is empty: Post length={post} Setting negative_word_ratio to zero'.format(
                    post=len(tokens)))
                negative_word_ratio = 0
            bad_word.append(negative_word_ratio)
            bad_words.append(bad_word_list)

        X_sub=np.array(bad_words)
        X = np.array([question, answer, anonymous, user, bad_word]).T
        X_final=np.hstack((X,X_sub))
        return X_final