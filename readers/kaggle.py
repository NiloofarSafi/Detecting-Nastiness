from __future__ import print_function, division
import pandas as pd
import numpy as np
import os
from model import Question, Answer, QuestionAnswer, Post
from sklearn.cross_validation import StratifiedShuffleSplit
from collections import OrderedDict
from random import randint
import random

def split_train_dev_test(X, y, train_per, dev_per, test_per):
    if (train_per + test_per > 1):
        print("Train  Test split should sum to one")
        return

    sss = StratifiedShuffleSplit(y, n_iter=1, test_size=test_per, random_state=1234)
    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]

    if dev_per > 0.0:
        sss_dev = StratifiedShuffleSplit(Y_train, n_iter=1, test_size=dev_per, random_state=1234)
        for train_index, test_index in sss_dev:
            X_train_, X_dev = X_train[train_index], X_train[test_index]
            Y_train_, Y_dev = Y_train[train_index], Y_train[test_index]
    else:
        X_dev, Y_dev = np.array([]), np.array([])
        X_train_, Y_train_ = X_train, Y_train

    ''' Under_sampling '''
    ''''
    negative_class = []
    positive_class = []

    for i in range(0,len(X_train_)):
        if Y_train_[i] == 'positive':
            positive_class.append([X_train_[i], Y_train_[i]])
        else:
            negative_class.append([X_train_[i], Y_train_[i]])

    rand = set([])
    while len(rand) < len(negative_class):
        rand.add(randint(0,len(positive_class)-1))

    print("random set is:")
    print(rand)
    print("the length of random set:")
    print(len(rand))
    print("the len of negative class:")
    print(len(negative_class))
    print("the len of positive class:")
    print(len(positive_class))


    for sample_number in rand:
        #print(sample_number)
        negative_class.append(positive_class[sample_number])

    print("len of negative class after adding positive class")
    print(len(negative_class))

    random.shuffle(negative_class)
    #random.shuffle(negative_class)

    X_train_new = []
    Y_train_new = []
    for n in negative_class:
        X_train_new.append(n[0])
        Y_train_new.append(n[1])'''

    ''' Over_sampling '''

    '''negative_class = []
    positive_class = []

    for i in range(0,len(X_train_)):
        if Y_train_[i] == 'positive':
            positive_class.append([X_train_[i], Y_train_[i]])
        else:
            negative_class.append([X_train_[i], Y_train_[i]])

    difference = len(positive_class) - len(negative_class)

    if difference < len(negative_class):

        rand = set([])
        while len(rand) < difference:
            rand.add(randint(0,difference))

        print("random set is:")
        print(rand)
        print("the length of random set:")
        print(len(rand))
        print("the len of negative class:")
        print(len(negative_class))
        print("the len of positive class:")
        print(len(positive_class))


        for sample_number in rand:
        #print(sample_number)
            negative_class.append(positive_class[sample_number])

        print("len of negative class after adding positive class")
        print(len(negative_class))

        random.shuffle(negative_class)
        #random.shuffle(negative_class)

        X_train_new = []
        Y_train_new = []
        for n in negative_class:
            X_train_new.append(n[0])
            Y_train_new.append(n[1])

    #else:'''


    return X_train_, Y_train_, X_dev, Y_dev, X_test, Y_test


class KaggleCorpus(object):
    _base_ask_dir = os.path.join(os.path.dirname(__file__), '..', "resources")

    def __init__(self, data_fname=None, training_data_fname=None, test_data_fname=None, dev_data_fname=None, train_per=0.7,
                 dev_per=0.1, test_per=0.3):
        self.train_per = train_per
        self.test_per = test_per
        self.dev_per = dev_per

        if data_fname and os.path.exists(os.path.join(self._base_ask_dir, data_fname)):

            self.corpus_path = os.path.join(self._base_ask_dir, data_fname)
            self.X, self.y = self.load_data()
            self._training_x, self._training_y, self._dev_x, self._dev_y, self._test_x, self._test_y = split_train_dev_test(
                self.X, self.y, train_per,
                dev_per, test_per)
        else:
            if training_data_fname and os.path.exists(os.path.join(self._base_ask_dir, training_data_fname)):
                self._training_x, self._training_y = self.load(os.path.join(self._base_ask_dir, training_data_fname))
            if test_data_fname and os.path.exists(os.path.join(self._base_ask_dir, test_data_fname)):
                self._test_x, self._test_y = self.load(os.path.join(self._base_ask_dir, test_data_fname))
            if dev_data_fname and os.path.exists(os.path.join(self._base_ask_dir, dev_data_fname)):
                self._dev_x, self._dev_y = self.load(os.path.join(self._base_ask_dir, dev_data_fname))

    def load_data(self):
        """
        question_id,bad_word,question,question_sentiment_gold,answer,answer_sentiment_gold
        :return:
        """
        X, y = [], []
        df = pd.read_csv(self.corpus_path, encoding='utf-8')
        for index, row in df.iterrows():
            # post = Post(id=row['Date'] + '_post', data=row['Comment'], actual_label=row['Insult'], tagged_data=row['pos_tag'])
            post = Post(id=row['rev_id'] + '_post', data=row['Comment'], actual_label=row['Insult'],
                        tagged_data=row['pos_tag'], SentLabel=row['Comment_label'])
            '''question = Question(anonymous="Anonymous", id=row['question_id'] + '_question', data=row['question'],
                                actual_label=row['question_sentiment_gold'], bad_word=row['bad_word'],
                                tagged_data=row['pos_tag_question'])
            answer = Answer(id=row['question_id'] + '_answer', data=row['answer'],
                            actual_label=row['answer_sentiment_gold'], bad_word=row['bad_word'],
                            tagged_data=row['pos_tag_answer'])
            qa = QuestionAnswer(question, answer)
            X.append(qa)
            y.append(qa.label)'''
            X.append(post)
            y.append(post.label)
        return np.array(X), np.array(y)

    def load(self, fname):
        X, y = [], []
        df = pd.read_csv(fname, encoding='utf-8')
        for index, row in df.iterrows():
            # post = Post(id=row['Date'], data=row['Comment'], actual_label=row['Insult'], tagged_data=row['pos_tag'],
            #             bad_word='')
            post = Post(id=row['rev_id'], data=row['Comment'], actual_label=row['Insult'], tagged_data=row['pos_tag'],
                        SentLabel=row['Comment_label'],bad_word='')
            '''if row['post_type'] == 'Question':
                question = Question(id=row['id'], data=row['post'],
                                    actual_label=row['label'], bad_word=row['bad_word'],
                                    tagged_data=row['pos_tag_question'])
                X.append(question)
                y.append(row['label'])
            elif row['post_type'] == 'Answer':
                answer = Answer(id=row['id'], data=row['post'],
                                actual_label=row['label'], bad_word=row['bad_word'], tagged_data=row['pos_tag_answer'])

                X.append(answer)
                y.append(row['label'])'''
            X.append(post)
            y.append(row['Insult'])
        return np.array(X), np.array(y)

    def _separate_qa(self, X, y):
        from readers import Bunch
        X_, y_ = [], []
        for xi, label in zip(X, y):
            '''X_.append(xi.question)
            y_.append(xi.question.label)
            X_.append(xi.answer)
            y_.append(xi.answer.label)'''
            X_.append(xi)
            y_.append(xi.label)

        return Bunch(instances=np.array(X_), labels=np.array(y_))

    @property
    def training_set_post(self):
        from readers import Bunch
        return Bunch(instances=self._training_x, labels=self._training_y)

    @property
    def dev_set_post(self):
        from readers import Bunch
        return Bunch(instances=self._dev_x, labels=self._dev_y)

    @property
    def test_set_post(self):
        from readers import Bunch
        return Bunch(instances=self._test_x, labels=self._test_y)

    @property
    def training_set(self):
        return self._separate_qa(self._training_x, self._training_y)

    @property
    def dev_set(self):
        return self._separate_qa(self._dev_x, self._dev_y)

    @property
    def test_set(self):
        return self._separate_qa(self._test_x, self._test_y)

    def save(self, fname, data='training'):
        """

        :param data: {training, training_post, test, test_post, dev, dev_post}
        :return:
        """
        data_type = dict(
            training=self.training_set,
            test=self.test_set,
            dev=self.dev_set,
            training_post=self.training_set_post,
            test_post=self.test_set_post,
            dev_post=self.dev_set_post
        )
        dataset = data_type.get(data, None)
        out_data = []
        if dataset:
            for xi, y in zip(dataset.instances, dataset.labels):
                row = OrderedDict(id=xi.id, post=xi.content, pos=xi.tagged_data, label=xi.label)
                '''if xi.post_type == 'Question':
                    row = OrderedDict(id=xi.id, bad_word=xi.bad_word, post=xi.content, pos=xi.tagged_data,
                                      label=xi.label,
                                      post_type='Question')
                if xi.post_type == 'Answer':
                    row = OrderedDict(id=xi.id, bad_word=xi.bad_word, post=xi.content, pos=xi.tagged_data,
                                      label=xi.label,
                                      post_type='Answer')
                if xi.post_type == 'QuestionAnswer':
                    row = OrderedDict(id=xi.question.id, bad_word=xi.question.bad_word, question=xi.question.content,
                                      question_sentiment_gold=xi.question.label, answer=xi.answer.content,
                                      answer_sentiment_gold=xi.answer.label, pos_tag_question=xi.question.tagged_data,
                                      pos_tag_answer=xi.answer.tagged_data, post_label=xi.label)'''

                out_data.append(row)
        cols=list(reversed(out_data[0].keys()))
        df = pd.DataFrame(out_data,columns=cols)
        df.to_csv(fname, encoding='utf-8', index=False)

    @staticmethod
    def load_as_pandas_df(fname):
        return pd.read_csv(fname, encoding='utf-8')


__all__ = ['KaggleCorpus']
