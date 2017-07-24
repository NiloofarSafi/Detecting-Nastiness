# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from . import lexical
from . import syntactic
from . import embeddings
from . import phonetic
from . import readability
from . import sentiments
from . import writing_density
from . import dumped_features
from . import badwords
from . import domain
from . import emotions
from . import LIWC
from . import patterns
from . import word2vec
from . import topics
from . import rnn
from . import doc2vec
import logging
import string
from itertools import chain
from preprocess import ark_tweet_tokenizer
from helpers.list_helpers import flattern

log = logging.getLogger(__name__)

__all__ = ['lexical', 'embeddings', 'phonetic', 'readability', 'writing_density', 'sentiments', 'get_feature',
           'create_feature', 'dumped_features', 'badwords', 'domain', 'emotions']


def preprocess(x):
    return x.replace('\n', ' ').replace('\r', '').replace('\x0C', '').lower()


def get_feature(f_name):
    """Factory to create features objects

    Parameters
    ----------
    f_name : features name

    Returns
    ----------
    features: BaseEstimator
        feture object

    """
    features_dic = dict(
        unigram=lexical.NGramTfidfVectorizer(ngram_range=(1, 1), tokenizer=ark_tweet_tokenizer, analyzer="word",
                                             lowercase=True, min_df=2),
        bigram=lexical.NGramTfidfVectorizer(ngram_range=(2, 2), tokenizer=ark_tweet_tokenizer, analyzer="word",
                                            lowercase=True, min_df=2),
        trigram=lexical.NGramTfidfVectorizer(ngram_range=(3, 3), tokenizer=ark_tweet_tokenizer, analyzer="word",
                                             lowercase=True, min_df=2),

        #
        # #char ngram
        char_tri=lexical.NGramTfidfVectorizer(ngram_range=(3, 3), analyzer="char",
                                              lowercase=True, min_df=2),
        char_4_gram=lexical.NGramTfidfVectorizer(ngram_range=(4, 4), analyzer="char", lowercase=True, min_df=2),

        char_5_gram=lexical.NGramTfidfVectorizer(ngram_range=(5, 5), analyzer="char", lowercase=True, min_df=2),

        # categorical character ngrams
        categorical_char_ngram_beg_punct=lexical.CategoricalCharNgramsVectorizer(beg_punct=True, ngram_range=(3, 3)),
        categorical_char_ngram_mid_punct=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3), mid_punct=True),
        categorical_char_ngram_end_punct=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3), end_punct=True),

        categorical_char_ngram_multi_word=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3), multi_word=True),
        categorical_char_ngram_whole_word=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3), whole_word=True),
        categorical_char_ngram_mid_word=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3), mid_word=True),

        categorical_char_ngram_space_prefix=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),
                                                                                    space_prefix=True),
        categorical_char_ngram_space_suffix=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),
                                                                                    space_suffix=True),

        categorical_char_ngram_prefix=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3), prefix=True),
        categorical_char_ngram_suffix=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3), suffix=True),

        # #skip gram
        two_skip_3_grams=lexical.KSkipNgramsVectorizer(k=2, tokenizer=ark_tweet_tokenizer, ngram=3, lowercase=True),
        two_skip_2_grams=lexical.KSkipNgramsVectorizer(k=2, tokenizer=ark_tweet_tokenizer, ngram=2, lowercase=True),

        # pos
        pos=syntactic.POSTags(ngram_range=(1, 1), tokenizer=string.split, analyzer="word", use_idf=False, norm='l1'),

        #pos_colored_ngrams
        pos_color_unigram=syntactic.WordPOSNGramTfidfVectorizer(ngram_range=(1, 1), tokenizer=string.split, analyzer="word",lowercase=True, min_df=2),
        pos_color_bigram=syntactic.WordPOSNGramTfidfVectorizer(ngram_range=(2, 2), tokenizer=string.split, analyzer="word",lowercase=True, min_df=2),
        pos_color_trigram=syntactic.WordPOSNGramTfidfVectorizer(ngram_range=(3, 3), tokenizer=string.split, analyzer="word",lowercase=True, min_df=2),

        emo_color_unigram = sentiments.SWNNGramTfidfVectorizer(ngram_range=(1, 1), tokenizer=string.split, analyzer="word",lowercase=True, min_df=2),
        emo_color_bigram=sentiments.SWNNGramTfidfVectorizer(ngram_range=(2, 2), tokenizer=string.split,
                                                             analyzer="word", lowercase=True, min_df=2),
        emo_color_trigram=sentiments.SWNNGramTfidfVectorizer(ngram_range=(3, 3), tokenizer=string.split,
                                                            analyzer="word", lowercase=True, min_df=2),
        # #phrasal and clausal
        phrasal=syntactic.Constituents(PHR=True),
        clausal=syntactic.Constituents(CLS=True),
        phr_cls=syntactic.Constituents(PHR=True, CLS=True),

        # #lexicalized and unlexicalized production rules
        lexicalized=syntactic.LexicalizedProduction(use_idf=False),
        unlexicalized=syntactic.UnLexicalizedProduction(use_idf=False),
        gp_lexicalized=syntactic.GrandParentLexicalizedProduction(use_idf=False),
        gp_unlexicalized=syntactic.GrandParentUnLexicalizedProduction(use_idf=False),

        # writing density
        wrd=Pipeline([('wr', writing_density.WritingDensityFeatures()), ('scaler', StandardScaler())]),

        # readability
        readability=readability.ReadabilityIndicesFeatures(),

        concepts=sentiments.SenticConceptsTfidfVectorizer(ngram_range=(1, 1), tokenizer=string.split, analyzer="word",
                                                          lowercase=True, binary=True, use_idf=False),

        concepts_score=sentiments.SenticConceptsScores(),

        google_word_emb=embeddings.Word2VecFeatures(tokenizer=ark_tweet_tokenizer, analyzer="word",
                                                    lowercase=True,
                                                    model_name='/home/niloofar/Shahryar_Niloofar/all/niloofar/Niloofar/NLP_final_project-Copy/resources/GoogleNews-vectors-negative300.bin.gz'),

        # phonetics
        phonetic=phonetic.PhoneticCharNgramsVectorizer(ngram_range=(3, 3), analyzer='char', min_df=2, lowercase=False),

        phonetic_scores=phonetic.PhonemeGroupBasedFeatures(),

        baseline_bad_words=badwords.BadWordRatio(),
        qa=domain.QuestionAnswerFeature(),

        swn=sentiments.SentiWordNetFeature(),
        emo=Pipeline([('emo', emotions.EmoticonsFeature()), ('scaler', StandardScaler())]),

        liwc = LIWC.LIWCFeature(),
        patterns = patterns.PatternFeature(),
        embedding = word2vec.Word2VecFeatures(),
        topics = topics.LDAFeatures(),
        rnn = rnn.RNNFeatures(),
        doc2vec = doc2vec.Doc2VecFeatures(model_name='/home/niloofar/Shahryar_Niloofar/WikiDocModelMain.model'),
        emotionbased=sentiments.EmotionFeature(ngram_range=(1, 1), tokenizer=string.split, analyzer="word",
                                               lowercase=True, min_df=2),
        liwcemotionbased=sentiments.LIWCEmotionFeature(ngram_range=(1, 1), tokenizer=ark_tweet_tokenizer, analyzer="word",
                                               lowercase=True, min_df=2)
    )

    return features_dic[f_name]


def create_feature(feature_names):
    """Utility function to create features object

    Parameters
    -----------
    feature_names : features name or list of features names


    Returns
    --------
    a tuple of (feature_name, features object)
       lst features names are joined by -
       features object is the union of all features in the lst

    """

    def feature_creater(f_names):
        try:
            if isinstance(f_names, list):
                return "-".join(f_names), FeatureUnion([(f, get_feature(f)) for f in f_names])
            else:

                return f_names, get_feature(f_names)
        except Exception as e:
            log.error("Error:: {} ".format(e))
            raise ValueError('Error in function ')

    if isinstance(feature_names, list):
        return  "-".join(flattern(feature_names)), FeatureUnion([(feature_creater(f)) for f in feature_names])

    else:
        return feature_creater(feature_names)
