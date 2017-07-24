# -*- coding: utf-8 -*-
from __future__ import print_function, division
from preprocess.normalize import apply_rules
import re

# Part-of-Speech Tagset
# N common noun
# O pronoun (personal/WH; not possessive)
# ^ proper noun
# S nominal + possessive
# Z proper noun + possessive
# V verb including copula, auxiliaries
# L nominal + verbal (e.g. i’m), verbal + nominal (let’s)
# M proper noun + verbal
# A adjective
# R adverb
# ! interjection
# D determiner
# P pre- or postposition, or subordinating conjunction
# & coordinating conjunction
# T verb particle
# X existential there, predeterminers
# Y X + verbal
# # hashtag (indicates topic/category for tweet)
# @ at-mention (indicates a user as a recipient of a tweet)
# ~ discourse marker, indications of continuation across
# multiple tweets
# U URL or email address
# E emoticon
# $ numeral
# , punctuation
# G other abbreviations, foreign words, possessive endings,
# symbols, garbage


#normalize rules

ASKFM={
   r'[@＠][a-zA-Z0-9_]+': "@username",  # @amb1213 -> @username
    r"((www\.[^\s]+)|(https?:\/\/[^\s]+))": "URL",  # url
    r"#([a-zA-Z0-9_]+)": r"\1",  # remove hashtag

}


normalize_askfm=lambda text:apply_rules(ASKFM)(text)



class Post(object):
    # def __init__(self, id, data, actual_label, bad_word,rnn_vector=None, predicted_label='', tagged_data=''):
    def __init__(self, id, data, actual_label, bad_word, rnn_vector=None, predicted_label='', tagged_data='',
                     SentLabel=''):
        self.id = id
        self.content = data
        self.label = actual_label
        self.bad_word = bad_word
        self.predicted_label = predicted_label
        self.tagged_data = tagged_data if tagged_data else 'EMPTY/A/0.7743'
        self.rnn_vector = rnn_vector
        self.SentLabel = SentLabel

##        self.post_type = None

    @property
    def post_type(self):
        return self.__class__.__name__

    @property
    def content(self):
        return self.__data

    @content.setter
    def content(self, data):
        # replace urls
        if data:
            data = data.strip()
            data = normalize_askfm(data)
            self.__data = data
        else:
            self.__data="EMPTY"

    @property
    def pos_tag(self, ):
        return " ".join([tag.rsplit('/', 2)[1] for tag in self.tagged_data.split()])

    @property
    def label(self):
        return self.__label

    @label.setter
    def label(self, label):
        label = label.strip().lower()
        if label == 'strong_negative':
            self.__label = 'negative'
            self.strength='strong'

        else:
            self.strength='normal'
            self.__label = label

    @property
    def emoticons(self):
        emoticons = []
        for tag in self.tagged_data.split():
            token, pos, confidence = tag.rsplit('/', 2)
            if pos == 'E':
                emoticons.append(token)
        return emoticons


    @property
    def wordpos(self):
        return " ".join([tag.rsplit('/', 1)[0] for tag in self.tagged_data.split()])



    def has_proper_noun(self):
        for tag in self.tagged_data.split():
            token, pos, confidence = tag.rsplit('/', 2)
            if pos == '^':
                return True
        return False

    def token_and_tag(self):
        lst = []
        for tag in self.tagged_data.split():
            # print(tag)
            # print("******^^^^^^******")
            token, pos, confidence = tag.rsplit('/', 2)
            lst.append([token, pos])
        return lst

    def __str__(self):
        return "id: %s, post: %s, label: %s, bad_word: %s type: %s pos_tags: %s rnn: %s sentLabel: %s" % (
            self.id, self.content, self.label, self.bad_word, self.post_type, self.tagged_data, self.rnn_vector,
            self.SentLabel)

        # self.id, self.content, self.label, self.bad_word, self.post_type, self.tagged_data, self.rnn_vector, self.SentLabel)



class Question(Post):
    def __init__(self, anonymous, **kwargs):
        #print("%%%%%%%%%%%%%%%%%%")
        self.anonymous = True if  anonymous == "Anonymous" else False
        super(Question, self).__init__(**kwargs)


    def is_defending(self):
        """Is the question defending the user by asking anonymous users to get out

        :return: Boolean
        """
        pass

    def is_anonymous(self):
        return self.anonymous


class Post(Post):
    def __init__(self, **kwargs):
        super(Post, self).__init__(**kwargs)

class Answer(Post):
    def __init__(self, **kwargs):
        super(Answer, self).__init__(**kwargs)


class QuestionAnswer(object):
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer
        self.label = "positive" if self.question.label == 'positive' and self.answer.label == 'positive' else "negative"


class User(object):
    def __init__(self, user_id, username, questions, likes, qa=None):
        self.user_id = user_id
        self.username = username
        self.questions = questions
        self.likes = likes
        self.qa = qa if qa else list()

    def add_qa(self, qa):
        self.qa.append(qa)


if __name__ == '__main__':
    p=Post( id='question_box_121238351987',data='Your fucking perfection',actual_label='positive',bad_word='ass',tagged_data='Your/D/0.7795 fucking/A/0.9443 perfection/N/0.9994')
    print(p.wordpos)