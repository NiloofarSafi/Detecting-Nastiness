# -*- coding: utf-8 -*-

__author__ = 'sjmaharjan'

# normalization regex rules
# https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions

CONTRACTIONS = {
    r"\bI'm\b": "I am",
    r"\bain't\b": "am not",
    r"\bwon't\b": "will not",
    r"\bshan't\b": "shall not",
    r"\b([a-z]+)n't\b": r"\1 not",  # for don't, doesn't, haven't, hasn't, ...for
    r"\b([a-z]+)'ve\b": r"\1 have",  # for 've -> have
    r"\b([a-z]+)'ll\b": r"\1 will",  # for 'll -> will
    r"\b([a-z]+)'re\b": r"\1 are",  # for 're -> are
}

SOCIAL_MEDIA = {
    r'[@＠][a-zA-Z0-9_]+': "@username",  # @amb1213 -> @username
    r"((www\.[^\s]+)|(https?:\/\/[^\s]+))": "URL",  # url
    r"#([a-zA-Z0-9_]+)": r"\1",  # remove hashtag
    r'\b2day\b': 'today',
    r'\b(2moro|2morrow|2morow|2mro)\b': 'tomorrow',
    r'\b4get\b': "forget",
    r'\b4got\b': "forgot",
    r'\bb4\b': "before",
    r'\b&\b': "and",  # &-> and
    r"\burs\b": "yours",
    r"\bur\b": "your",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bsth\b": "something",
    r"\bsb\b": "somebody",

}






token_patterns = r'(?:[A-Za-z]\.)+' \
                 r'|(?:\$?\d+(?:\,\d+)*(?:\.\d+)?)' \
                 r"|\w+(?:[-']\w+)*" \
                 r'''|[.,;"'?():-_`] ''' \
                 r"|[-.()]+" \
                 r"|\[" \
                 r"|\S\w*"
