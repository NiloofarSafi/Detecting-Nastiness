# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.base import BaseEstimator,TransformerMixin
import re
import numpy as np
import nltk
from preprocess import ark_tweet_tokenizer

__all__ = ['paragraphs', 'WritingDensityFeatures']


def paragraphs(document):
    """Helper method to divide document to paragraphs

    Paragraph is defined by punctuation followed by a new line

    Parameters
    ------------
    docuemnt : string


    Retruns
    ----------
    paragraphs : list of paragraphs

    """
    punctuation = '''!"'().?[]`{}'''
    paragraph = re.compile(r'[{}]\n'.format(re.escape(punctuation)))
    return paragraph.split(document)


class WritingDensityFeatures(BaseEstimator,TransformerMixin):
    def get_feature_names(self):
        return np.array(
            ['n_words', 'n_chars', 'exclamation', 'question', 'avgwordlenght', 'avesentencelength',
             'avgwordspersentence', 'allcaps', 'diversity'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        exclamation = [doc.content.count('!') if doc.content else 0 for doc in documents]
        # print exclamation
        question = [doc.content.count('?') if doc.content else 0 for doc in documents]
        # print question

        n_words = [len(ark_tweet_tokenizer(doc.content)) if doc.content else 0 for doc in documents]
        # print n_words
        n_chars = [len(doc.content) if doc.content else 0 for doc in documents]

        avg_sent_length = [np.mean([len(sent) for sent in ark_tweet_tokenizer(doc.content)]) if doc.content else 0 for doc in documents]

        avg_word_lenght = [np.mean([len(word) for word in ark_tweet_tokenizer(doc.content)]) if doc.content else 0 for doc in documents]

        avg_words_sentence = [np.mean([len(ark_tweet_tokenizer(sent)) for sent in nltk.sent_tokenize(doc.content)]) if doc.content else 0 for
                              doc in documents]

        all_caps = [np.sum([word.isupper() for word in ark_tweet_tokenizer(doc.content) if len(word) > 2]) if doc.content else 0   for doc in
                    documents]

        # avg_sent_per_paragraph = [np.mean([len(nltk.sent_tokenize(par + ".")) for par in paragraphs(doc.content)]) for
        #                           doc in documents]

        diversity = [(len(set(ark_tweet_tokenizer(doc.content))) * 1.0) / (len(ark_tweet_tokenizer(doc.content)) * 1.0) if doc.content else 0
                     for doc in
                     documents]

        X = np.array(
            [n_words, n_chars, exclamation, question, avg_word_lenght, avg_sent_length, avg_words_sentence, all_caps,
             diversity]).T
        return X

