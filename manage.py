# -*- coding: utf-8 -*-

from __future__ import print_function
import click
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from config import config
import os
import sys
from celery import group
from helpers.file_helpers import stdout_redirector
import logging


logging.basicConfig( stream=sys.stderr, level=logging.DEBUG )
# Set the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

app = config[os.getenv('CYBERBULLYING_CONFIG') or 'default']




@click.group()
def manager():
    pass


@manager.command()
@click.option('--count', default=1, help='number of greetings')
@click.argument('name')
def classification(count, name):
    print("Inside the command")
    print(count)
    print(name)


@manager.command()
@click.option('--input', help='/path/to/input/file')
@click.option('--output', help='/path/to/output/file')
def postag(input, output):
    from readers.askfm import AskfmCorpus
    from tagger.CMUTweetTagger import runtagger_parse
    from model import normalize_askfm
    df = AskfmCorpus.load_as_pandas_df(input)
    # print("I passed the panda.load")
    df.fillna("EMPTY", inplace=True)

    # tag question and answer
    df['pos_tag_question'] = [" ".join(["/".join([t, p, str(c)]) for t, p, c in tagged_post]) for tagged_post in
                              runtagger_parse(df.question.apply(normalize_askfm).values)]
    df['pos_tag_answer'] = [" ".join(["/".join([t, p, str(c)]) for t, p, c in tagged_post]) for tagged_post in
                            runtagger_parse(df.answer.apply(normalize_askfm).values)]

    # only answer are empty
    df['answer'][df['answer'] == 'EMPTY'] = " "
    df['pos_tag_answer'][df['pos_tag_answer'] == 'EMPTY/A/0.7743'] = " "

    # save to file
    df.to_csv(output, encoding='utf-8', index=False)


@manager.command()
@click.option('--corpus', help='corpus file')
@click.option('--training', help='training data percentage', type=float)
@click.option('--dev', help='development data percentage [taken from training data]', type=float)
@click.option('--output', help='/path/to/output.dir')
def preparedata(corpus, training, dev, output):
    from readers.askfm import AskfmCorpus
    corpus = AskfmCorpus(data_fname=corpus, train_per=training, dev_per=dev, test_per=1.0 - training)
    corpus.save(os.path.join(output, 'training.csv'), data='training')
    if dev >0:
        corpus.save(os.path.join(output, 'dev.csv'), data='dev')
    corpus.save(os.path.join(output, 'test.csv'), data='test')



@manager.command()
# @click.option('--corpus', help='corpus file')
@click.option('--train', help='train set file')
@click.option('--dev', help='dev set file')
@click.option('--test', help='test set file')
# def runbaseline(corpus):
def runbaseline(train, dev, test):
    # from readers.askfm import AskfmCorpus
    from readers.kaggle import KaggleCorpus
    from experiments.base import Experiment
    from features import create_feature
    from classifiers import baselines
    training_per = 0.7
    dev_per = 0.2
    parameters = {'T': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}
    # corpus = AskfmCorpus(data_fname=corpus, train_per=training_per, dev_per=dev_per, test_per=1.0 - training_per)
    corpus = KaggleCorpus(training_data_fname=train, dev_data_fname=dev, test_data_fname=test)
    feature = create_feature('baseline_bad_words')
    classifier = baselines.BaselineBadWordRatioClassifier()

    with open(os.path.join('/home/niloofar/Niloofar/all/niloofar/Niloofar/suraj/cyberbullying/output/ACL/wiki', 'wiki_bad_words_baseline.txt'), 'w') as f:
        with stdout_redirector(f):
            Experiment('Baseline_', corpus.training_set, corpus.dev_set, corpus.test_set)(classifier=classifier,
                                                                                          parameters=parameters,
                                                                                          features=feature)



@manager.command()
@click.option('--corpus', help='corpus file')
@click.option('--feature', help='feature name')
def testexp(corpus,feature):
    from readers.askfm import AskfmCorpus
    from experiments.base import Experiment

    from features import create_feature

    training_per = 0.7
    dev_per = 0.2
    parameters = {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]}

    corpus = AskfmCorpus(data_fname=corpus, train_per=training_per, dev_per=dev_per, test_per=1.0 - training_per)
    feature_obj = create_feature(feature)
    classifier =LinearSVC(class_weight='balanced')

    with open(os.path.join(app.OUT_DIRECTORY, 'Test_'+feature+'.txt'), 'w') as f:
        with stdout_redirector(f):
            Experiment('Test_', corpus.training_set, corpus.dev_set, corpus.test_set)(classifier=classifier,
                                                                                          parameters=parameters,
                                                                                          features=feature_obj)


@manager.command()
@click.option('--corpus', help='/path/to/corpus/file')
@click.option('--classifier', help='[SVM|LR]')
def runexp(corpus, classifier):
    from tasks import run_experiment
    jobs = group(run_experiment('Experiment_', corpus, feature, classifier) for feature in app.FEATURES)
    jobs.apply_async()



@manager.command()
@click.option('--corpus', help='/path/to/corpus/file')
# @click.option('--train', help='train set file')
# @click.option('--dev', help='dev set file')
# @click.option('--test', help='test set file')
# @click.option('--rnn_train', help='rnn training set file')
#@click.option('--rnn_test', help='rnn test set file')
@click.option('--classifier', help='[SVM|LR]')
def runexpser(corpus, classifier):
# def runexpser(train, dev, test, classifier):
    from tasks import run_experiment
    for feature in app.FEATURES:
        print ("Feature : %s"%feature)
        # run_experiment('Experiment_', train, dev, test, feature, classifier)
        run_experiment('Experiment_', corpus, feature, classifier)
        print ("done")



@manager.command()
@click.option('--input', help='/path/to/corpus/file')
def display_result(input):
    from utils import display_classification_results
    display_classification_results(input)

if __name__ == "__main__":
    manager()
