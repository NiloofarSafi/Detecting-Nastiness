# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from matplotlib import pylab
import os

from sklearn.pipeline import Pipeline

from manage import app

CHART_DIR=app.CHART_DIR


def show_most_informative_features(vectorizer, clf, n=20):
    if isinstance(vectorizer,Pipeline):

        c_f = sorted(zip(clf.coef_[0], vectorizer.steps[0][1].get_feature_names()))
    else:
        c_f = sorted(zip(clf.coef_[0], vectorizer.get_feature_names()))
    top = list(zip(c_f[:n], c_f[:-(n + 1):-1]))
    print("\t%s\t%-15s\t\t%s\t%-15s" % ('Score','Top Negative', 'Score', 'Top Positive'))
    for (c1, f1), (c2, f2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1, f1.encode('utf-8'), c2, f2.encode('utf-8')))





def plot_feat_importance(name,classifier, vectorizer, n_top_features=25):
    # get coefficients with large absolute values
    if isinstance(vectorizer, Pipeline):
        feature_names=vectorizer.steps[0][1].get_feature_names()
    else:
        feature_names=vectorizer.get_feature_names()
    if len(feature_names) <n_top_features:
        n_top_features=len(feature_names)

    coef = classifier.coef_.ravel()
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])
    # plot them
    pylab.figure(num=None, figsize=(15, 5))

    colors = ["red" if c < 0 else "blue" for c in coef[interesting_coefficients]]
    pylab.bar(np.arange(2*n_top_features), coef[interesting_coefficients], color=colors)
    feature_names = np.array(feature_names)
    pylab.title('Feature importance for %s' % (name))
    pylab.xticks(np.arange(1, 2*n_top_features+1), feature_names[interesting_coefficients], rotation=70, ha="right")
    filename = name.replace(" ", "_")
    pylab.savefig(os.path.join(
        '/home/niloofar/Niloofar/all/niloofar/Niloofar/suraj/cyberbullying/output/ACL/ask/Analysis', "wiki_feat_imp_%s.png" % filename), bbox_inches="tight")
