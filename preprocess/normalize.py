# -*- coding: utf-8 -*-
__author__ = 'sjmaharjan'

from rules import CONTRACTIONS, SOCIAL_MEDIA
import re


def build_normalization_rules(search, replace):
    return lambda text: re.sub(search, replace, text, flags=re.IGNORECASE)


def rules(rule_dic):
    return map(lambda (search, replace): build_normalization_rules(search, replace), rule_dic.iteritems())


def apply_rules(rule_dict):
    def apply(text):
        for rule in rules(rule_dict):
            text = rule(text)
        return text

    return apply


def contraction(text):
    return apply_rules(CONTRACTIONS)(text)


def social_media(text):
    return apply_rules(SOCIAL_MEDIA)(text)



#
# normalize_corpus1 = lambda text: corpus1_rules(contraction(text))
# normalize_corpus2 = lambda text: corpus2_rules(social_media(contraction(text)))
# normalize_corpus3 = lambda text: corpus3_rules(social_media(contraction(text)))