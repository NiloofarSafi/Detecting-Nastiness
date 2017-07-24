from __future__ import division, print_function
from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
from preprocess import ark_tweet_tokenizer
import codecs
import os

__all__=['PatternFeature']

pattern = ["L D A N", "L D A A N", "L R D N", "L R D A N", "L R D A A N", "V O", "O N", "N N", "N N N", "O A N", "V D N"]
You = ["you", "your", "youre", "you're"]


class PatternFeature(BaseEstimator, TransformerMixin):
    """
    cyberbullying pattern Feature estimator
    """
    _bad_words = os.path.join(os.path.dirname(__file__), 'resources')

    def __init__(self, bad_word_file='bad_word.txt', pat=None):
        self._bad_word_file = bad_word_file
        self._load_bad_words(os.path.join(self._bad_words, self._bad_word_file))
        self.You = You
        if not pat:
            self.pattern_list = pattern
        else:
            self.pattern_list = pat

    def _load_bad_words(self, file):
        bad_word_list = set()
        with codecs.open(file, mode='r') as f:
            for line in f:
                bad_word_list.add(line.strip(" \r\n"))
        self.bad_word_list_ = list(set(bad_word_list))

    def get_feature_names(self):
        return np.array(['pattern1', "pattern2", "pattern3" , "pattern4", "pattern5", "pattern6",
                         "pattern7", "pattern8", "pattern9", "pattern10", "pattern11", "total pattern"])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        pat1, pat2, pat3, pat4, pat5, pat6, pat7, pat8, pat9, pat10, pat11, total_pattern = [], [], [], [], [], [], [], [], [], [], [], []

        for doc in documents:
            p1 = 0
            p2 = 0
            p3 = 0
            p4 = 0
            p5 = 0
            p6 = 0
            p7 = 0
            p8 = 0
            p9 = 0
            p10 = 0
            p11 = 0
            total = 0
            for y in self.You:
                if y in doc.content:
                    toktaglist = doc.token_and_tag()
                    tokens = ark_tweet_tokenizer(doc.content)
                    if self.pattern_list[0] in doc.pos_tag:
                        for i in range(0, (len(toktaglist)-3)):
                            if toktaglist[i][0] in self.You and toktaglist[i+1][1] == 'D':
                                for word in self.bad_word_list_:
                                    if word in [toktaglist[i+2][0], toktaglist[i+3][0]]  :
                                        p1 = 1
                                        total += 1
                    if self.pattern_list[1] in doc.pos_tag:
                        for i in range(0, (len(toktaglist)-4)):
                            if toktaglist[i][0] in self.You and toktaglist[i+1][1] == 'D':
                                for word in self.bad_word_list_:
                                    if word in [toktaglist[i+2][0],toktaglist[i+3][0],toktaglist[i+4][0]]:
                                        p2 = 1
                                        total += 1
                    if self.pattern_list[2] in doc.pos_tag:
                        for i in range(0, (len(toktaglist)-3)):
                            if toktaglist[i][0] in self.You:
                                if toktaglist[i+1][1] == 'R' and toktaglist[i+2][1] == 'D':
                                    for word in self.bad_word_list_:
                                        if word in toktaglist[i+3][0]:
                                            p3 = 1
                                            total += 1
                    if self.pattern_list[3] in doc.pos_tag:
                        for i in range(0, (len(toktaglist)-4)):
                            if toktaglist[i][0] in self.You:
                                if toktaglist[i+1][1] == 'R' and toktaglist[i+2][1] == 'D':
                                    for word in self.bad_word_list_:
                                        if word in [toktaglist[i+3][0], toktaglist[i+4][0]]:
                                            p4 = 1
                                            total += 1
                    if self.pattern_list[4] in doc.pos_tag:
                        for i in range(0, (len(toktaglist)-5)):
                            if toktaglist[i][0] in self.You:
                                if toktaglist[i+1][1] == 'R' and toktaglist[i+2][1] == 'D':
                                    for word in self.bad_word_list_:
                                        if word in [toktaglist[i+3][0], toktaglist[i+4][0], toktaglist[i+5][0]]:
                                            p5 = 1
                                            total += 1
                    if self.pattern_list[5] in doc.pos_tag:
                        for i in range(1, (len(toktaglist))):
                            if toktaglist[i][0] in self.You:
                                if toktaglist[i-1][1] == 'V':
                                    for word in self.bad_word_list_:
                                        if word in toktaglist[i-1][0]:
                                            p6 = 1
                                            total += 1
                    if self.pattern_list[6] in doc.pos_tag:
                        for i in range(0, (len(toktaglist)-1)):
                            if toktaglist[i][0] in self.You:
                                if toktaglist[i+1][1] == 'N':
                                    for word in self.bad_word_list_:
                                        if word in toktaglist[i+1][0]:
                                            p7 = 1
                                            total += 1
                    if self.pattern_list[9] in doc.pos_tag:
                        for i in range(0, (len(toktaglist)-2)):
                            if toktaglist[i][0] in self.You:
                                if toktaglist[i+1][1] == 'A' and toktaglist[i+2][1] == 'N':
                                    for word in self.bad_word_list_:
                                        if word in toktaglist[i+2][0]:
                                            p10 = 1
                                            total += 1
                else:
                    toktaglist = doc.token_and_tag()
                    if self.pattern_list[7] in doc.pos_tag:
                        for i in range(0, (len(toktaglist)-1)):
                            if toktaglist[i][1] == 'N' and toktaglist[i+1][1] == 'N':
                                counter = 0
                                for word in self.bad_word_list_:
                                    if word in [toktaglist[i][0], toktaglist[i+1][0]]:
                                        counter += 1
                                if counter == 2:
                                    p8 = 1
                                    total += 1
                    if self.pattern_list[8] in doc.pos_tag:
                        for i in range(0, (len(toktaglist)-2)):
                            if toktaglist[i][1] == 'N' and toktaglist[i+1][1] == 'N' and toktaglist[i+2][1] == 'N':
                                counter = 0
                                for word in self.bad_word_list_:
                                    if word in [toktaglist[i][0], toktaglist[i+1][0], toktaglist[i+2][0]]:
                                        counter += 1
                                if counter == 2:
                                    p9 = 1
                                    total += 1
                    if self.pattern_list[10] in doc.pos_tag:
                        for i in range(0, (len(toktaglist)-2)):
                            if toktaglist[i][1] == 'V'and toktaglist[i+1][1] == 'D' and toktaglist[i+2][1] == 'N':
                                counter = 0
                                for word in self.bad_word_list_:
                                    if word in [toktaglist[i][0], toktaglist[i+2][0]]:
                                        counter += 1
                                if counter == 2:
                                    p11 = 1
                                    total += 1
            pat1.append(p1)
            pat2.append(p2)
            pat3.append(p3)
            pat4.append(p4)
            pat5.append(p5)
            pat6.append(p6)
            pat7.append(p7)
            pat8.append(p8)
            pat9.append(p9)
            pat10.append(p10)
            pat11.append(p11)
            total_pattern.append(round(total/11, 2))

        X = np.array([pat1, pat2, pat3, pat4, pat5, pat6, pat7, pat8, pat9, pat10, pat11, total_pattern]).T
        return X
