# -*- coding: utf-8 -*-
from __future__ import print_function
import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    DATA = os.path.join(basedir,'resources','Data_05_16_2016.csv')
    OUT_DIRECTORY=os.path.join(basedir,'output','analysis')
    CELERY_BROKER_URL = "amqp://guest:guest@localhost:5672//"
    CELERY_BACKEND_URL = "amqp://guest:guest@localhost:5672//"
    CHART_DIR=os.path.join(basedir,'charts')



    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    FEATURES = [

        # Lexical Features
        # word ngrams
        #'topics',
        'unigram'
        #'bigram',
        #'trigram',
       #  'unigram',
       #  'bigram',
       #  'trigram',
        #  'unigram',
        #  'bigram',
        #  'trigram',
       # #
         #['unigram', 'trigram'],
         #['unigram', 'bigram'],
         #['bigram', 'trigram'],
         #['unigram', 'bigram', 'trigram'],
       #   ['unigram', 'bigram', 'trigram', 'rnn'],
       # #
       # # #  # char ngrams
        #'emotionbased',
        # 'liwcemotionbased',
         #'char_tri',
         #'char_4_gram',
         #'char_5_gram',
         #['char_tri', 'char_4_gram', 'char_5_gram'],
         #['char_tri', 'char_4_gram', 'char_5_gram', 'unigram'],
         #['char_tri', 'char_4_gram', 'char_5_gram', 'swn'],
         # ['char_tri', 'char_4_gram', 'char_5_gram', 'liwc'],
         # ['char_tri', 'char_4_gram', 'char_5_gram', 'topics'],
         # ['char_tri', 'char_4_gram', 'char_5_gram', 'rnn'],
       # # #
       # # #  # Typed char ngrams
        # 'categorical_char_ngram_beg_punct',
        # 'categorical_char_ngram_mid_punct',
        # 'categorical_char_ngram_end_punct',
        # 'categorical_char_ngram_multi_word',
        # 'categorical_char_ngram_whole_word',
        # 'categorical_char_ngram_mid_word',
        # 'categorical_char_ngram_space_prefix',
        # 'categorical_char_ngram_space_suffix',
        # 'categorical_char_ngram_prefix',
        # 'categorical_char_ngram_suffix',
       # #
       #
        #'emotionbased',
         #['unigram', 'emotionbased'],
         # ['unigram', 'liwcemotionbased'],
         #['emotionbased', 'liwc', 'emo', 'swn'],
         #['emotionbased', 'liwc', 'emo', 'swn', 'embedding', 'doc2vec'],
         #['emotionbased', 'topics'],
         #['emotionbased', 'pos_color_unigram'],
         #['emotionbased', 'embedding'],
         #['emotionbased', 'doc2vec'],
        ##['emotionbased', 'liwc', 'emo', 'swn', 'topics'],
        ##['emotionbased', 'liwc', 'emo', 'swn', 'unigram'],
        ##['emotionbased', 'liwc', 'emo', 'swn', 'char_tri', 'char_4_gram', 'char_5_gram'],
        ##['unigram', 'doc2vec', 'embedding'],
        ##['char_tri', 'char_4_gram', 'char_5_gram', 'doc2vec','embedding'],
        ##['pos_color_unigram','unigram'],
        ##['pos_color_unigram', 'pos_color_bigram', 'pos_color_trigram', 'unigram', 'bigram', 'trigram'],
        ##['unigram', 'bigram', 'trigram', 'char_tri', 'char_4_gram', 'char_5_gram'],
        ##['unigram', 'char_4_gram', 'pos_color_unigram', 'qa', 'doc2vec', 'topics'],
        ##['unigram', 'char_4_gram', 'qa', 'emo'],
        ##['emotionbased', 'liwc', 'emo', 'swn', 'embedding', 'doc2vec', 'unigram', 'char_4_gram', 'qa']
        # ['emotionbased', 'liwc', 'emo', 'swn', 'doc2vec', 'embedding'],
        # ['doc2vec', 'patterns', 'topics', 'swn'],
        # ['pos_color_unigram', 'pos_color_bigram', 'pos_color_trigram', 'doc2vec'],
        # ['char_4_gram', 'unigram', 'emo', 'topics', 'doc2vec']
         #['emotionbased', 'doc2vec', 'embedding'],
        #'liwcemotionbased',
         # 'pos_color_unigram',
         # 'pos_color_bigram',
         # 'pos_color_trigram',
         # ['unigram', 'liwc', 'qa', 'patterns', 'swn','char_tri', 'char_4_gram', 'char_5_gram', 'topics' ],
        ##['unigram', 'liwc', 'qa', 'patterns','char_tri', 'char_4_gram', 'char_5_gram', 'topics' ],
         # ['unigram', 'liwc', 'qa', 'patterns', 'topics'],
         # ['unigram', 'liwc', 'qa', 'patterns', 'rnn'],
         # ['unigram', 'liwc', 'qa', 'patterns','char_tri', 'char_4_gram', 'char_5_gram', 'rnn'],
         # ['unigram', 'qa', 'char_tri', 'char_4_gram', 'char_5_gram', 'topics' ],
         # ['unigram', 'qa', 'char_tri', 'char_4_gram', 'char_5_gram', 'rnn'],
         # ['unigram', 'qa', 'liwc','patterns','char_tri', 'char_4_gram', 'char_5_gram', 'topics' ],
         # ['unigram', 'qa', 'liwc', 'topics'],
         # ['unigram', 'qa', 'liwc', 'rnn'],
        ##['unigram', 'topics'],
         # ['unigram', 'rnn'],
        ##['char_tri', 'char_4_gram', 'char_5_gram', 'topics' ],
         # ['char_tri', 'char_4_gram', 'char_5_gram', 'rnn' ],
         # ['unigram', 'bigram', 'trigram', 'topics'],
       # #  # Syntactic Features
         #'pos_color_unigram',
         #'pos_color_bigram',
         #'pos_color_trigram',
         #['pos_color_unigram', 'pos_color_bigram', 'pos_color_trigram'],
       #   ['pos_color_unigram', 'pos_color_bigram', 'pos_color_trigram', 'topics'],
       #   ['pos_color_unigram', 'pos_color_bigram', 'pos_color_trigram', 'rnn'],
        #'emo_color_unigram',
        #'emo_color_bigram',
        #'emo_color_trigram'
       # #  'pos',
       # #  'phrasal',
       # #  'clausal',
       # #  'phr_cls',
       # #  'lexicalized',
       # #  # 'unlexicalized',
       # #  'gp_lexicalized',
       # #  'gp_unlexicalized',
       # #
       # #  # WR and Readability
       #    'wrd',
       #  'readability',
        #'wrd',
        # 'readability',
       # #
       # #
       # # # #Phonetic Features
       # #  'phonetic',
       # #  'phonetic_scores',
       #
       # #  #sentic concepts and scores
       # #   ['concepts_score','concepts']
         # 'emo',
         # 'qa',
          #'swn',
       #   ['qa', 'emo'],
       #   ['qa', 'rnn'],
        ##['qa', 'liwc'],
       #   ['qa', 'embedding'],
       #   ['qa', 'liwc', 'topics'],
       #   ['qa', 'topics'],
        ##['topics', 'liwc'],
       #   ['topics', 'rnn'],
       #   ['embedding', 'rnn'],
       #   ['patterns', 'topics', 'embedding', 'rnn'],
       #   ['unigram', 'qa', 'liwc', 'patterns', 'rnn'],
        ##['patterns', 'topics'],
        ##['qa', 'patterns', 'liwc'],
       #   ['qa', 'unigram', 'embedding', 'topics'],
       #   ['char_tri', 'char_4_gram', 'char_5_gram', 'swn'],
       #   ['char_tri', 'char_4_gram', 'char_5_gram', 'qa', 'unigram', 'embedding', 'topics'],
       #   ['char_tri', 'rnn'],
       #   ['char_tri', 'char_4_gram', 'char_5_gram', 'qa', 'unigram', 'embedding', 'topics', 'rnn'],
       #   ['unigram', 'char_tri', 'rnn'],
       #   ['unigram', 'liwc', 'patterns', 'rnn'],
        ##['unigram', 'char_tri'],
        ##['unigram', 'char_4_gram'],
       #   ['unigram', 'liwc', 'patterns', 'emo'],
       #   ['unigram', 'liwc', 'patterns', 'emo', 'char_tri', 'topics', 'qa'],
       #   ['unigram', 'liwc', 'patterns', 'emo', 'char_tri', 'topics', 'qa', 'rnn'],
       #   ['unigram', 'liwc', 'patterns', 'swn', 'char_tri', 'topics', 'qa'],
       #   ['unigram', 'liwc', 'patterns', 'wrd', 'char_tri', 'topics', 'qa'],
       #   ['char_tri', 'pos_color_unigram', 'pos_color_bigram', 'pos_color_trigram', 'liwc'],
       #   ['unigram', 'liwc', 'patterns', 'emo', 'embedding', 'topics'],
       #   ['unigram', 'liwc', 'patterns', 'emo', 'embedding', 'topics', 'rnn'],
       #   ['unigram', 'liwc', 'patterns', 'emo', 'char_tri', 'topics'],
       #   ['unigram', 'liwc', 'patterns', 'swn', 'char_tri' , 'topics'],
       #   ['unigram', 'liwc', 'patterns', 'wrd', 'char_tri', 'topics'],
       #   ['char_tri', 'pos_color_unigram', 'pos_color_bigram', 'pos_color_trigram', 'topics'],
        ##['embedding', 'topics'],
       #   ['embedding', 'topics', 'rnn'],
        ##['char_tri', 'embedding'],
        ##['char_tri', 'swn'],
        ##['char_tri', 'liwc'],
        ##['char_tri', 'topics'],
       #   ['char_tri', 'qa'],
        ##['char_tri', 'emo'],
        ##['char_tri', 'patterns'],
        ##['char_tri', 'liwc', 'emo', 'embedding'],
       #   ['char_tri', 'liwc', 'emo', 'embedding', 'rnn', 'qa'],
       #   ['char_tri', 'liwc', 'emo', 'rnn', 'qa'],
       #   ['char_tri', 'liwc', 'swn', 'qa'],
        ##['char_tri', 'liwc', 'emo', 'embedding', 'topics'],
       #   ['char_tri', 'liwc', 'emo', 'embedding', 'topics', 'rnn'],
       #   ['char_tri', 'liwc', 'swn', 'qa', 'topics'],
       #   ['wrd', 'rnn'],
        ##['wrd', 'topics'],
        ##['char_tri', 'two_skip_2_grams', 'two_skip_3_grams'],
       #   ['unigram', 'bigram', 'trigram', 'qa', 'pos_color_unigram', 'pos_color_bigram', 'pos_color_trigram', 'liwc', 'topics',
       #   'patterns', 'char_tri', 'char_4_gram', 'char_5_gram','embedding', 'swn', 'two_skip_2_grams', 'two_skip_3_grams', 'rnn'],
       #   ['unigram', 'liwc', 'patterns', 'qa', 'emo', 'char_tri', 'char_4_gram', 'char_5_gram','two_skip_2_grams', 'topics'],
       #   ['unigram', 'char_tri', 'char_4_gram', 'char_5_gram', 'embedding', 'two_skip_2_grams', 'qa', 'topics'],
                   ##['unigram', 'bigram', 'trigram', 'pos_color_unigram', 'pos_color_bigram', 'pos_color_trigram', 'liwc', 'topics', 'emo', 'wrd', 'embedding', 'doc2vec',
          ##'patterns', 'char_tri', 'char_4_gram', 'char_5_gram','embedding', 'swn', 'two_skip_2_grams', 'two_skip_3_grams', 'emotionbased', 'liwcemotionbased', 'qa'],

         #['unigram', 'bigram', 'trigram', 'pos_color_unigram', 'pos_color_bigram', 'pos_color_trigram', 'liwc', 'topics', 'emo', 'wrd', 'embedding', 'doc2vec',
         # 'patterns', 'char_tri', 'char_4_gram', 'char_5_gram', 'embedding', 'swn', 'two_skip_2_grams', 'two_skip_3_grams', 'emotionbased', 'liwcemotionbased'],

        # #  # #  # Google w2v embedding
       #  'google_word_emb',
       #   'two_skip_3_grams',
       #   'two_skip_2_grams',
       #   ['two_skip_3_grams', 'two_skip_2_grams'],
        ##'liwc',
        ##'topics',
        ##'patterns',
        ##['embedding', 'doc2vec'],
       #   'rnn'
       #   'baseline_bad_words'
        ##'doc2vec',
        ##'embedding',
          #['unigram', 'doc2vec'],
          #['unigram', 'bigram', 'trigram', 'doc2vec'],
          #['doc2vec', 'topics'],
          #['doc2vec', 'qa'],
        ##['doc2vec', 'qa', 'unigram', 'swn', 'emo', 'topics'],
          #['doc2vec', 'char_4_gram', 'unigram', 'qa', 'two_skip_2_grams'],
          # ['doc2vec', 'embedding'],
        ##['doc2vec', 'patterns', 'topics', 'swn'],
        ##['pos_color_unigram', 'pos_color_bigram', 'pos_color_trigram', 'doc2vec'],

       #  #
       #   ['char_4_gram', 'rnn'],
       #   ['char_4_gram', 'rnn', 'unigram'],
        ##['qa','emo'],
       #   ['swn','char_4_gram'],
       #   ['unigram','wrd']
       #   ['char_4_gram','unigram'],
       #   ['char_4_gram','unigram','qa'],
       #   ['char_4_gram','unigram','emo'],
       #   ['char_4_gram','unigram','swn'],
       #   ['char_4_gram','unigram','wrd'],
        #  ['char_4_gram','unigram','qa','categorical_char_ngram_prefix'],
        # ['char_4_gram','unigram','qa','categorical_char_ngram_suffix'],
        #
         # ['char_4_gram','unigram','rnn','two_skip_2_grams'],
         # ['char_4_gram','unigram','two_skip_2_grams', 'embedding', 'rnn'],
         # ['char_4_gram','unigram','two_skip_2_grams', 'liwc', 'rnn'],
         # ['char_4_gram','unigram','rnn','two_skip_2_grams', 'embedding', 'topics'],
         # ['char_4_gram','unigram','wrd','two_skip_2_grams'],
        # ['char_4_gram','unigram','qa','wrd','two_skip_2_grams','categorical_char_ngram_suffix'],
        # ['char_4_gram','unigram','qa','wrd','two_skip_2_grams','categorical_char_ngram_prefix'],
        #  ['qa','emo'],
        #  ['swn','char_4_gram'],
        #  ['unigram','wrd'],
        #  ['char_4_gram','unigram'],
        #  ['char_4_gram','unigram','qa', 'topics'],
        #  ['char_4_gram','unigram','qa', 'rnn'],
        #  ['char_4_gram','unigram','qa', 'topics', 'rnn'],
        #  ['char_4_gram','unigram','qa', 'topics', 'embedding', 'rnn'],
        #  ['liwc','embedding','qa', 'topics', 'rnn'],
        #  ['liwc','embedding','topics', 'rnn'],
        #  ['liwc','embedding','patterns', 'topics', 'rnn'],
        ##['char_4_gram','unigram','qa','emo', 'topics'],
        #  ['char_4_gram','unigram','qa','swn', 'topics'],
        #  ['char_4_gram','unigram','qa','wrd'],
        # ['char_4_gram','unigram','qa','categorical_char_ngram_prefix'],
        # ['char_4_gram','unigram','qa','categorical_char_ngram_suffix'],
        #
         # ['char_4_gram','unigram','qa','wrd','two_skip_2_grams'],
        # ['char_4_gram','unigram','qa','wrd','two_skip_2_grams','categorical_char_ngram_suffix'],
        # ['char_4_gram','unigram','qa','wrd','two_skip_2_grams','categorical_char_ngram_prefix'],
        #  ['qa','emo'],
        #  ['liwc','patterns'],
        #  ['liwc','patterns','qa'],
        #  ['liwc','patterns','unigram','char_4_gram'],
        #  ['liwc','patterns', 'topics'],
        #  ['liwc','patterns','embedding', 'topics'],
        #  ['liwc','patterns','qa', 'topics'],
        #  ['liwc','patterns','unigram','char_4_gram', 'topics'],
         #['liwc','patterns','qa','unigram','char_4_gram', 'categorical_char_ngram_prefix'],
         #['liwc', 'patterns', 'qa', 'char_4_gram','unigram','categorical_char_ngram_suffix'],
         # ['liwc','patterns', 'embedding'],
         # ['liwc','patterns','unigram','char_4_gram','embedding'],
         # ['liwc','patterns','unigram','char_4_gram','embedding', 'topics'],
         # #['liwc','patterns','qa','unigram','char_4_gram', 'categorical_char_ngram_prefix', 'embedding'],
         # ['char_4_gram','unigram','emo','swn','liwc','patterns'],
         # ['char_4_gram','unigram','emo','swn','liwc','patterns', 'topics', 'qa'],
         # ['char_4_gram','unigram','emo','swn'],
         # ['char_4_gram','unigram','emo','swn','embedding'],
         # ['topics', 'swn'],
         # ['topics', 'emo'],
         # ['liwc','patterns','unigram', 'char_tri', 'char_4_gram', 'char_5_gram', 'qa', 'rnn']
         #['liwc', 'patterns', 'qa', 'unigram', 'char_4_gram', 'two_skip_2_grams', 'categorical_char_ngram_suffix'],
         #['liwc', 'patterns', 'qa', 'unigram', 'char_4_gram', 'two_skip_2_grams', 'categorical_char_ngram_prefix'],
         #['liwc', 'patterns', 'qa', 'unigram', 'char_4_gram', 'two_skip_2_grams', 'categorical_char_ngram_suffix', 'embedding'],
         #['liwc', 'patterns', 'qa', 'unigram', 'char_4_gram', 'two_skip_2_grams', 'categorical_char_ngram_prefix', 'embedding'],

         # ['char_4_gram', 'swn'],
        ##['unigram','wrd'],
        ##['char_4_gram', 'unigram', 'emo', 'topics', 'embedding'],
        ##['char_4_gram', 'unigram', 'emo', 'topics', 'doc2vec'],
        ##['char_4_gram','unigram']
         # ['char_4_gram','unigram','qa'],
         # ['char_4_gram','unigram','qa','emo'],
         # ['char_4_gram','unigram','qa','swn'],
         # ['char_4_gram','unigram','qa','wrd'],
         #['char_4_gram','unigram','qa','categorical_char_ngram_prefix'],
         #['char_4_gram','unigram','qa','categorical_char_ngram_suffix'],
        ##
         # ['char_4_gram','unigram','qa','wrd','two_skip_2_grams']
         #['char_4_gram','unigram','qa','wrd','two_skip_2_grams','categorical_char_ngram_suffix'],
         #['char_4_gram','unigram','qa','wrd','two_skip_2_grams','categorical_char_ngram_prefix']

    ]

    VECTORS='.'



class TestingConfig(Config):
    TESTING = True


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
