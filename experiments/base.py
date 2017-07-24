# -*- coding: utf-8 -*-
from __future__ import print_function
import logging
import random
from sklearn import metrics

from sklearn.cross_validation import PredefinedSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from analysis import feature_imp
import analysis
import random
from random import randint

# Set up logger
log = logging.getLogger(__name__)


class Experiment(object):
    def __init__(self, experiment_name, training_set=None, dev_set=None, test_set=None):
        """
        initialize the experiment
        initializes the name,  training, dev and test data
        """
        self._name = experiment_name
        self._training_set = training_set
        self._dev_set = dev_set
        self._test_set = test_set


    def get_name(self):
        return self._name

    def __call__(self, **kwargs):
        """
        template method
        train and then test on test instances
        finally prints the evaluation report
        """

        try:

            model, features_obj, le = self.train(**kwargs)

            X_test, y_test = self._test_set.instances, self._test_set.labels

            X_test_features = features_obj.transform(X_test)

            y_test_labels = le.transform(y_test)

            print()
            print("[INFO] Test Set  instances %d and labels %d" % (
                X_test.shape[0], y_test.shape[0]))

            y_predicted = model.predict(X_test_features)
            score = model.score(X_test_features, y_test_labels)

            self.report(le, y_test_labels, y_predicted)

            fea_imp=kwargs.get('feature_imp',False)
            if fea_imp:
                feature_imp.show_most_informative_features(features_obj,model,n=50)
                feature_imp.plot_feat_importance(name=self.get_name(),classifier=model,vectorizer=features_obj,n_top_features=25)

            mistakes=kwargs.get('mistake',None)
            if mistakes:
                analysis.show_mistakes(mistakes,X_test,y_test_labels,y_predicted)





        except Exception as ex:
            log.error('Error running the experiment. {error}s'.format(
                      error=ex))
            return False
        else:
            self.post_process(model, features_obj, y_predicted)

        return score

    # TODO for cases that don't have dev set, cv for training set depending on the parameter
    def train(self, **kwargs):
        # get the feature object and the classifier
        print ("Experiment: %s"%self.get_name())
        feature_name, feature_obj = kwargs.get('features', (None, None))
        clf = kwargs.get('classifier', None)
        parameters = kwargs.get('parameters', {})
        class_le = LabelEncoder()

        print ("Feature: %s"%feature_name)


        # X_train_rnn = np.load("E:\\suraj\\cyberbullying\\resources\\train_test\\X_train.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
        # Y_train_rnn = np.load("E:\\suraj\\cyberbullying\\resources\\train_test\\y_train.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
        # print("hereeeeeeeeeeeeeeeeeeeeee")
        # print(X_train_rnn)
        # print(Y_train_rnn)
        # print(X_train_rnn[0])
        # print(Y_train_rnn[0])
        '''pca = PCA(n_components=2)
        pca.fit(X_train_rnn)
        fig = plt.figure()
        transformed = pca.transform(X_train_rnn)
        indices_X = np.where(Y_train_rnn == 0)[0]
        sub1 = []
        sub2 = []
        for i in xrange(len(transformed)):
            #plt.scatter(transformed[i][0],transformed[i][1],c='blue',alpha=0.3,edgecolors='black')
            if i in indices_X:
                sub1.append(transformed[i])
                #plt.scatter(transformed[i][0],transformed[i][1],c='blue',alpha=0.3,edgecolors='black')
            else:
                sub2.append(transformed[i])
                #plt.scatter(X_train_rnn[i][0],X_train_rnn[i][1],c='red',alpha=0.3,edgecolors='black')
        sub1 = np.array(sub1)
        sub2 = np.array(sub2)
        plt.scatter(sub1[:,0],sub1[:,1],c='blue',alpha=0.3,edgecolors='black') #negative
        plt.scatter(sub2[:,0],sub2[:,1],c='red',alpha=0.3,edgecolors='black') #positive
        print("I am here")
        plt.legend()
        plt.grid(True)
        plt.title("RNN trained vectors in 2D space")
        plt.show()
        fig.savefig('C:\\Users\\Niloofar Safi\\Desktop\\RNN_trained_vectors.png',dpi=fig.dpi)'''



        if self._dev_set:
            X_train, X_dev = self._training_set.instances, self._dev_set.instances


            y_train, y_dev = class_le.fit_transform(self._training_set.labels), class_le.transform(self._dev_set.labels)

            # concatenate the train and dev
            X_train_all, y_train_all = np.hstack((X_train, X_dev)), np.hstack((y_train, y_dev))

            #print(y_train_all)
            ''' Under_sampling '''

            '''negative_class = []
            positive_class = []

            for i in range(0,len(X_train_all)):
                if y_train_all[i] == 1:
                    positive_class.append([X_train_all[i], y_train_all[i]])
                else:
                    negative_class.append([X_train_all[i], y_train_all[i]])

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

            X_train_new = np.ndarray(shape=(0,))
            y_train_new = np.ndarray(shape=(0,))
            for n in negative_class:
                X_train_new = np.append(X_train_new, n[0])
                y_train_new = np.append(y_train_new, n[1])'''

            ''' Over_sampling '''
            # print("I am hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
            negative_class = []
            positive_class = []

            for i in range(0,len(X_train_all)):
                if y_train_all[i] == 1:
                    positive_class.append([X_train_all[i], y_train_all[i]])
                else:
                    negative_class.append([X_train_all[i], y_train_all[i]])

            difference = len(positive_class) - len(negative_class)
            print(difference)

            rand = []
            random.seed(0)
            while len(rand) < difference:

                rand.append(randint(0, len(negative_class)-1))

            # print("random set is:")
            # print(rand)
            print("the length of random set:")
            print(len(rand))
            print("the len of negative class:")
            print(len(negative_class))
            print("the len of positive class:")
            print(len(positive_class))

            for sample in negative_class:
                positive_class.append(sample)

            for sample_number in rand:
                #print(sample_number)
                positive_class.append(negative_class[sample_number])

            print("len of positive class after adding positive class")
            print(len(positive_class))

            random.shuffle(positive_class)
                #random.shuffle(negative_class)

            X_train_new = []
            y_train_new = []
            for n in positive_class:
                X_train_new = np.append(X_train_new, n[0])
                y_train_new = np.append(y_train_new, n[1])



            print("[INFO] Training instances %d and labels %d" % (X_train.shape[0], y_train.shape[0]))
            print("[INFO] Classes are : %s" % ", ".join(list(class_le.classes_)))
            print("[INFO] Validation set instances %d and labels %d" % (X_dev.shape[0], y_dev.shape[0]))


            # perform grid search over the parameter values and find the best classifier on the dev set
            print(X_train)
            print(y_train)
            X_train_features = feature_obj.fit_transform(X_train, y_train)
            # print("hereeeeeeeeeeeeeeeeeeeeeeeee")
            X_dev_features = feature_obj.transform(X_dev)
            # print("hereeeeeeeeeeeeeeeeeeeeeeeee")
            #X_train_all_features = feature_obj.transform(X_train_new, y_train_new)


            print ("Shape of X_train: {}".format(X_train_features.shape))
            print ("Shape of X_dev: {}".format(X_dev_features.shape))
            #print ("Shape of X_train_rnn: {}".format(X_train_rnn.shape))

            #print ("Shape of X_train_all: {}".format(X_train_new_features.shape))

            if sp.issparse(X_train_features):
                X_train_all_features = sp.vstack((X_train_features, X_dev_features))

            else:

                X_train_all_features = np.vstack((X_train_features, X_dev_features))


            print ("Shape of X_train + X_dev: {}".format(X_train_all_features.shape))



            # predefined split all training instances marked by -1 and test instances by 0

            ps = PredefinedSplit(test_fold=[-1] * X_train_features.shape[0] + [0] * X_dev_features.shape[0])

            grid = GridSearchCV(estimator=clf, param_grid=parameters, cv=ps,scoring='roc_auc')
            grid.fit(X_train_all_features, y_train_all)

            print("Best score: %0.3f" % grid.best_score_)
            print("Best parameters set:")
            best_parameters = grid.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))

            for params, mean_score, scores in grid.grid_scores_:
                print("%0.3f+/-%0.2f %r"
                      % (mean_score, scores.std() / 2, params))

            # train with best parameter
            X_train_features = feature_obj.fit_transform(X_train_new, y_train_new)
            best_clf = grid.best_estimator_.fit(X_train_features, y_train_new)

            print("[INFO] Training + Validations set instances %d and labels %d" % (
                X_train_new.shape[0], y_train_new.shape[0]))

            print("Training Accuracy =%.3f" % best_clf.score(X_train_features, y_train_new))

            return best_clf, feature_obj, class_le

        else:
            X_train, X_test = self._training_set.instances, self._test_set.instances
            y_train, y_test = class_le.fit_transform(self._training_set.labels), class_le.transform(self._test_set.labels)


    def report(self, le, y_test, y_pred):
        """
        prints the precision, recall, f-score
        print confusion matrix
        print accuracy


        """
        # pred = np.zeros_like(target)
        # for train, test in kf:
        #     met.fit(data[train], target[train])
        #     pred[test] = met.predict(data[test])
        #  rmse = np.sqrt(mean_squared_error(target, pred))

        print('---------------------------------------------------------')
        print()
        print("Classifation Report")
        print()

        target_names = le.classes_
        class_indices = {cls: idx for idx, cls in enumerate(le.classes_)}

        print(metrics.classification_report(y_test, y_pred, target_names=target_names,
                                            labels=[class_indices[cls] for cls in target_names]))

        print("============================================================")
        print("Confusion matrix")
        print("============================================================")
        print(target_names)
        print()
        print(confusion_matrix(
            y_test,
            y_pred,
            labels=[class_indices[cls] for cls in target_names]))

        print()

        precisions_micro, recalls_micro, fscore_micro, _ = precision_recall_fscore_support(y_test, y_pred,
                                                                                           average='micro',
                                                                                           pos_label=None)
        precisions_macro, recalls_macro, fscore_macro, _ = precision_recall_fscore_support(y_test, y_pred,
                                                                                           average='macro',
                                                                                           pos_label=None)
        precisions_weighted, recalls_weighted, fscore_weighted, _ = precision_recall_fscore_support(y_test, y_pred,
                                                                                                    average='weighted',
                                                                                                    pos_label=None)

        print()
        print('Test Accuracy: %.3f' % accuracy_score(y_test, y_pred))


        print("Macro Precision Score, %f, Micro Precision Score, %f, Weighted Precision Score, %f" % (
            precisions_macro, precisions_micro, precisions_weighted))

        print("Macro Recall score, %f, Micro Recall Score, %f, Weighted Recall Score, %f" % (
            recalls_macro, recalls_micro, recalls_weighted))

        print("Macro F1-score, %f, Micro F1-Score, %f, Weighted F1-Score, %f" % (
            fscore_macro, fscore_micro, fscore_weighted))

        print('Misclassified samples: %d' % (y_test != y_pred).sum())

        print('ROC AUC: %.3f' % roc_auc_score(y_true=y_test, y_score=y_pred))

        print("============================================================")

    def post_process(self, model, features, y_predicted):
        pass
