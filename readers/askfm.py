from __future__ import print_function, division
import pandas as pd
import numpy as np
import os
from model import Question, Answer, QuestionAnswer
from sklearn.cross_validation import StratifiedShuffleSplit
from collections import OrderedDict
from random import randint
import random

def split_train_dev(X, y, train_per, dev_per):
    if (train_per + dev_per > 1):
        print("Train Dev split should sum to one")
        return
    if dev_per>0.0:
        sss = StratifiedShuffleSplit(y, n_iter=1, test_size=train_per, random_state=1234)
        for train_index, test_index in sss:
            X_train, X_dev = X[train_index], X[test_index]
            Y_train, Y_dev = y[train_index], y[test_index]

    else:
        X_dev, Y_dev = np.array([]), np.array([])
        X_train, Y_train = X, y

    return X_train, Y_train, X_dev, Y_dev


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
            print(X_train[0])
            X_train_, X_dev = X_train[train_index], X_train[test_index]
            Y_train_, Y_dev = Y_train[train_index], Y_train[test_index]
    else:
        X_dev, Y_dev = np.array([]), np.array([])
        X_train_, Y_train_ = X_train, Y_train

    return X_train_, Y_train_, X_dev, Y_dev, X_test, Y_test


class AskfmCorpus(object):
    _base_ask_dir = os.path.join(os.path.dirname(__file__), '..', "resources")

    def __init__(self, data_fname=None, training_data_fname=None, test_data_fname=None, dev_data_fname=None, train_per=0.7,
                 dev_per=0.1, test_per=0.3, rnn_training_fname = None, rnn_test_fname = None, IsDev = False):
        self.train_per = train_per
        self.test_per = test_per
        self.dev_per = dev_per
        if data_fname and os.path.exists(os.path.join(self._base_ask_dir, data_fname)):

            self.corpus_path = os.path.join(self._base_ask_dir, data_fname)
            self.X, self.y = self.load_data()
            print(self.X.shape)
            self._training_x, self._training_y, self._dev_x, self._dev_y, self._test_x, self._test_y = split_train_dev_test(
                self.X, self.y, train_per,
                dev_per, test_per)
            #row_pos = ['train_pos']
            #row_neg = ['train_neg']
            #r_pos = ['test_pos']
            #r_neg = ['test_neg']
            #g = 0
            #for ins in self._training_x:
            #    #g += 2
            #    if ins.question.label == 'positive':
            #        g += 1
            #        row_pos.append(ins.question.content)
            #    if ins.answer.label == 'positive':
            #        g += 1
            #        row_pos.append(ins.answer.content)
            #    if ins.question.label == 'negative':
            #        g += 1
            #        row_neg.append(ins.question.content)
            #    if ins.answer.label == 'negative':
            #        g += 1
            #        row_neg.append(ins.answer.content)


            #for ins in self._dev_x:

            #    if ins.question.label == 'positive':
            #        g += 1
            #        row_pos.append(ins.question.content)
            #    if ins.answer.label == 'positive':
            #        g += 1
            #        row_pos.append(ins.answer.content)
            #    if ins.question.label == 'negative':
            #        g += 1
            #        row_neg.append(ins.question.content)
            #    if ins.answer.label == 'negative':
            #        g += 1
            ##        row_neg.append(ins.answer.content)

            #h = 0
            #for ins in self._test_x:

            #    if ins.question.label == 'positive':
            #        h += 1
            #        r_pos.append(ins.question.content)
            #    if ins.answer.label == 'positive':
            #        h += 1
            ##        r_pos.append(ins.answer.content)
            #    if ins.question.label == 'negative':
            #        h += 1
            #        r_neg.append(ins.question.content)
            #    if ins.answer.label == 'negative':
            #        h += 1
            #        r_neg.append(ins.answer.content)
            #print(g)
            #print(h)
            #df = pd.DataFrame(row_pos)
            #df2 = pd.DataFrame(row_neg)
            #df3 = pd.DataFrame(r_pos)
            #df4 = pd.DataFrame(r_neg)

            #df.to_csv(
            #    '/home/niloofar/Shahryar_Niloofar/all/niloofar/Niloofar/suraj/cyberbullying/resources/Pastor_data/train_pos.csv',
            #    index=False, header=False)
            ##df2.to_csv(
            #    '/home/niloofar/Shahryar_Niloofar/all/niloofar/Niloofar/suraj/cyberbullying/resources/Pastor_data/train_neg.csv',
            #    index=False, header=False)
            #df3.to_csv(
            #    '/home/niloofar/Shahryar_Niloofar/all/niloofar/Niloofar/suraj/cyberbullying/resources/Pastor_data/test_pos.csv',
            #    index=False, header=False)
            #df4.to_csv(
            #    '/home/niloofar/Shahryar_Niloofar/all/niloofar/Niloofar/suraj/cyberbullying/resources/Pastor_data/test_neg.csv',
            #    index=False, header=False)

            #print("askfmcorpus1")

        elif IsDev :
            if training_data_fname and rnn_training_fname and os.path.exists(os.path.join(self._base_ask_dir, training_data_fname)):
                self._train_x, self._train_y = self.load(os.path.join(self._base_ask_dir, training_data_fname),
                                                         rnn_training_fname, "E:\\suraj\\cyberbullying\\resources\\train_test\\y_train.npy")
                self._training_x, self._training_y, self._dev_x, self._dev_y = split_train_dev(self._train_x, self._train_y, self.train_per,
                                                                                                    self.dev_per)
            if test_data_fname and rnn_test_fname and os.path.exists(os.path.join(self._base_ask_dir, test_data_fname)):
                self._test_x, self._test_y = self.load((os.path.join(self._base_ask_dir, test_data_fname))
                                                        , rnn_test_fname, "E:\\suraj\\cyberbullying\\resources\\train_test\\y_test.npy")

        else:
            
            if training_data_fname and rnn_training_fname and os.path.exists(os.path.join(self._base_ask_dir, training_data_fname)):
                self._training_x, self._training_y = self.load(os.path.join(self._base_ask_dir, training_data_fname))
            if test_data_fname and rnn_test_fname and os.path.exists(os.path.join(self._base_ask_dir, test_data_fname)):
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
            question = Question(anonymous="Anonymous", id=row['question_id'] + '_question', data=row['question'],
                                actual_label=row['question_sentiment_gold'], bad_word=row['bad_word'],
                                tagged_data=row['pos_tag_question'], SentLabel=row['question_label'])
            answer = Answer(id=row['question_id'] + '_answer', data=row['answer'],
                            actual_label=row['answer_sentiment_gold'], bad_word=row['bad_word'],
                            tagged_data=row['pos_tag_answer'], SentLabel=row['answer_label'])
            # question = Question(anonymous="Anonymous", id=row['question_id'] + '_question', data=row['question'],
            #                     actual_label=row['question_sentiment_gold'], bad_word=row['bad_word'],
            #                     tagged_data=row['pos_tag_question'])
        
            # answer = Answer(id=row['question_id'] + '_answer', data=row['answer'],
            #                 actual_label=row['answer_sentiment_gold'], bad_word=row['bad_word'],
            #                 tagged_data=row['pos_tag_answer'])

            qa = QuestionAnswer(question, answer)
            X.append(qa)
            y.append(qa.label)
        return np.array(X), np.array(y)

    def load(self, fname, rnn, yrnn):
        X, y = [], []
        print("*************************************************************")
        print("--------------------------------------------------------------")
        print("**************************************************************")
        df = pd.read_csv(fname, encoding='utf-8')
        rnn_vec = np.load(rnn, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
        y_rnn = np.load(yrnn, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
        rnn_counter = 0
        print("The labels for rnn vectors")
        for i in range(len(y_rnn)):
            print(y_rnn[i])
        print("The labels for posts")
        for index,row in df.iterrows():
            print(row['label'])
        for index, row in df.iterrows():
            if row['post_type'] == 'Question':
                #print("**************")
                question = Question(anonymous="Anonymous", id=row['id'], data=row['post'],
                                    actual_label=row['label'], rnn_vector=rnn_vec[rnn_counter], bad_word=row['bad_word'],
                                    tagged_data=row['pos'], SentLabel=row['question_label'])
                #print(y_rnn[rnn_counter])
                rnn_counter += 1
                X.append(question)
                y.append(row['label'])
            elif row['post_type'] == 'Answer':
                answer = Answer(id=row['id'], data=row['post'],
                                actual_label=row['label'], bad_word=row['bad_word'],
                                tagged_data=row['pos'], rnn_vector=rnn_vec[rnn_counter], SentLabel=row['answer_label'])
                #print(y_rnn[rnn_counter])
                rnn_counter += 1
                X.append(answer)
                y.append(row['label'])

        return np.array(X), np.array(y)

    def _separate_qa(self, X, y):
        from readers import Bunch
        X_, y_ = [], []
        for xi, label in zip(X, y):
            X_.append(xi.question)
            y_.append(xi.question.label)
            X_.append(xi.answer)
            y_.append(xi.answer.label)
            # X_.append(xi)
            # y_.append(xi.label)

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
                if xi.post_type == 'Question':
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
                                      pos_tag_answer=xi.answer.tagged_data, post_label=xi.label)

                out_data.append(row)
        cols=list(reversed(out_data[0].keys()))
        df = pd.DataFrame(out_data,columns=cols)
        df.to_csv(fname, encoding='utf-8', index=False)

    @staticmethod
    def load_as_pandas_df(fname):
        return pd.read_csv(fname, encoding='utf-8')


__all__ = ['AskfmCorpus']
