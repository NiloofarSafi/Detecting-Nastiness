# -*- coding: utf-8 -*-
from __future__ import print_function, division
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import os
from manage import app
from features import create_feature
# from celery import Celery
from readers.askfm import AskfmCorpus
from readers.kaggle import KaggleCorpus
from experiments.base import Experiment
from helpers.file_helpers import stdout_redirector


# celery = Celery(__name__, backend=app.CELERY_BACKEND_URL, broker=app.CELERY_BROKER_URL)
#
# @celery.task
# def run_experiment(name, fname, feature_names, classifier):
def run_experiment(name, fname, feature_names, classifier):
    # def run_experiment(name, train, dev, test, feature_names, classifier):
    training_per = 0.7
    dev_per = 0.2
    parameters = {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]}
    # corpus = KaggleCorpus(training_data_fname=train, dev_data_fname=dev, test_data_fname=test)
    print("*******************************************************************")
    corpus = AskfmCorpus(data_fname=fname, train_per=training_per, dev_per=dev_per, test_per=1.0 - training_per)
    # corpus = AskfmCorpus(training_data_fname=train, test_data_fname=test, train_per=training_per, dev_per=dev_per,
    #                      rnn_training_fname= "E:\\suraj\\cyberbullying\\resources\\train_test\\X_train.npy",
    #                      rnn_test_fname= "E:\\suraj\\cyberbullying\\resources\\train_test\\X_test.npy", IsDev=True)
    #                      # test_per=1.0 - training_per)

    if classifier == 'SVM':
        clf = LinearSVC(class_weight='balanced')
    elif classifier == 'LR':
        clf = LogisticRegression(class_weight='balanced')

    feature = create_feature(feature_names)
    feature_output = "-".join(feature_names) if isinstance(feature_names, list) else feature_names
    with open(os.path.join('/home/niloofar/Niloofar/all/niloofar/Niloofar/suraj/cyberbullying/output/ACL/ask',
                           'ask_' + feature_output + '_' + classifier + '.txt'),
              'w') as f:
        with stdout_redirector(f):
            Experiment(name + feature_output + '_' + classifier, corpus.training_set, corpus.dev_set, corpus.test_set)(
                classifier=clf,
                parameters=parameters,
                features=feature,
                feature_imp=True,
                mistake=os.path.join(
                    '/home/niloofar/Niloofar/all/niloofar/Niloofar/suraj/cyberbullying/output/ACL/ask',
                    'Mistakes',
                    'ask_' + feature_output + '_' + classifier + '_mistake' + '.csv'))

