# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.feature_extraction.text import TfidfVectorizer

from collections import defaultdict
import csv
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import os, sys

reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')

__all__ = ['SentiWordNetFeature', 'EmotionFeature', 'LIWCEmotionFeature', 'SenticConceptsTfidfVectorizer', 'SenticConceptsScores']


# REF http://sentiwordnet.isti.cnr.it/code/SentiWordNetDemoCode.java
# REF Building Machine Learning Systems with Python Section Sentiment analysis
def load_sentiwordnet(path):
    scores = defaultdict(list)
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for line in reader:
            # skip comments
            if line[0].startswith("#"):
                continue
            if len(line) == 1:
                continue
            POS, ID, PosScore, NegScore, SynsetTerms, Gloss = line
            if len(POS) == 0 or len(ID) == 0:
                continue
            # print POS,PosScore,NegScore,SynsetTerms
            for term in SynsetTerms.split(" "):
                # drop number at the end of every term
                term = term.split("#")[0]
                term = term.replace("-", " ").replace("_", " ")
                key = "%s/%s" % (POS, term.split("#")[0])
                scores[key].append((float(PosScore), float(NegScore)))
    for key, value in scores.items():
        scores[key] = np.mean(value, axis=0)
    return scores


# REF Building Machine Learning Systems with Python Section Sentiment analysis
class SentiWordNetFeature(BaseEstimator,TransformerMixin):
    """
    Senti word net Feature estimator
    """
    _sentiwordnet_resource= os.path.join(os.path.dirname(__file__), 'resources')
    def __init__(self):
        self.sentiwordnet = load_sentiwordnet(os.path.join(self._sentiwordnet_resource,'SentiWordNet_3.0.0_20130122.txt'))

    def get_feature_names(self):
        return np.array(['sent_neut', 'sent_pos', 'sent_neg', 'nouns', 'adjectives', 'verbs', 'adverbs'])

    def _get_sentiments(self, d):
        tagged_sent = d.tagged_data
        pos_vals = []
        neg_vals = []
        nouns = 0.
        adjectives = 0.
        verbs = 0.
        adverbs = 0.
        sent_len = 0
        for tag in tagged_sent.split():

            t, p, c = tag.rsplit('/', 2)
            # print (t,p,c)
            p_val, n_val = 0, 0
            sent_pos_type = None
            if p.startswith("N") or p.startswith("^"):
                sent_pos_type = "n"
                nouns += 1
            elif p.startswith("A"):
                sent_pos_type = "a"
                adjectives += 1
            elif p.startswith("V") or p.startswith("T"):
                sent_pos_type = "v"
                verbs += 1
            elif p.startswith("R"):
                sent_pos_type = "r"
                adverbs += 1
            if sent_pos_type is not None:
                sent_word = "%s/%s" % (sent_pos_type, t.lower())
                if sent_word in self.sentiwordnet:
                    p_val, n_val = self.sentiwordnet[sent_word]
            pos_vals.append(p_val)
            neg_vals.append(n_val)
            sent_len += 1

        l = sent_len if sent_len else 1.0
        avg_pos_val = np.mean(pos_vals) if pos_vals else 0
        avg_neg_val = np.mean(neg_vals) if neg_vals else 0
        return [1 - avg_pos_val - avg_neg_val, avg_pos_val, avg_neg_val, nouns / l, adjectives / l, verbs / l,
                adverbs / l]

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        X = np.array([self._get_sentiments(d) for d in documents])
        return X

    # Emotion Features

class SWNNGramTfidfVectorizer(TfidfVectorizer):
    """Convert a collection of  documents objects to a matrix of TF-IDF features.

      Refer to super class documentation for further information
      POS coloring
      REF:http://www.anthology.aclweb.org/N/N12/N12-1.pdf#page=694
    """

    def build_analyzer(self):
        """Overrides the super class method

        Parameter
        ----------
        self

        Returns
        ----------
        analyzer : function
            extract content from document object and then applies analyzer

        """
        analyzer = super(TfidfVectorizer,
                         self).build_analyzer()
        return lambda doc: (w for w in analyzer(doc.wordemo))

class EmotionFeature(TfidfVectorizer):
        """
        Emotion feature estimator
        """
        _sentiwordnet_resource = os.path.join(os.path.dirname(__file__), 'resources')

        # def __init__(self):
        #    self.sentiwordnet = load_sentiwordnet(os.path.join(self._sentiwordnet_resource,'SentiWordNet_3.0.0_20130122.txt'))

        def build_analyzer(self):
            analyzer = super(TfidfVectorizer,
                             self).build_analyzer()
            self.sentiwordnet = load_sentiwordnet(
                os.path.join(self._sentiwordnet_resource, 'SentiWordNet_3.0.0_20130122.txt'))
            # lambda doc: (self._get_sentiments(w) for w in analyzer(doc.tagged_data))
            return lambda doc: (self._get_sentiments(w) for w in analyzer(doc.tagged_data))

        def _get_sentiments(self, w):
            # print("I am here")
            t, p, c = w.rsplit('/', 2)
            # print (t,p,c)
            p_val, n_val, neutral_val = 0, 0, 0
            sent_pos_type = None
            if p.startswith("N") or p.startswith("^"):
                sent_pos_type = "n"

            elif p.startswith("A"):
                sent_pos_type = "a"

            elif p.startswith("V") or p.startswith("T"):
                sent_pos_type = "v"

            elif p.startswith("R"):
                sent_pos_type = "r"

            if sent_pos_type is not None:
                sent_word = "%s/%s" % (sent_pos_type, t.lower())
                if sent_word in self.sentiwordnet:
                    p_val, n_val = self.sentiwordnet[sent_word]
                    neutral_val = 1 - p_val - n_val
                    if p_val > n_val:
                        if p_val > neutral_val:
                            return t+"/positive"
                        else:
                            return t+"/neutral"
                    elif n_val > p_val:
                        if n_val > neutral_val:
                            return t+"/negative"
                        else:
                            return t+"/neutral"
                    elif n_val == p_val:
                        return t+"/neutral"
                else:
                    return t+"/none"

                class EmotionFeature(TfidfVectorizer):
                    """
                    Emotion feature estimator
                    """
                    _sentiwordnet_resource = os.path.join(os.path.dirname(__file__), 'resources')

                    # def __init__(self):
                    #    self.sentiwordnet = load_sentiwordnet(os.path.join(self._sentiwordnet_resource,'SentiWordNet_3.0.0_20130122.txt'))

                    def build_analyzer(self):
                        analyzer = super(TfidfVectorizer,
                                         self).build_analyzer()
                        self.sentiwordnet = load_sentiwordnet(
                            os.path.join(self._sentiwordnet_resource, 'SentiWordNet_3.0.0_20130122.txt'))
                        # lambda doc: (self._get_sentiments(w) for w in analyzer(doc.tagged_data))
                        return lambda doc: (self._get_sentiments(w) for w in analyzer(doc.tagged_data))

                    def _get_sentiments(self, w):
                        # print("I am here")
                        t, p, c = w.rsplit('/', 2)
                        # print (t,p,c)
                        p_val, n_val, neutral_val = 0, 0, 0
                        sent_pos_type = None
                        if p.startswith("N") or p.startswith("^"):
                            sent_pos_type = "n"

                        elif p.startswith("A"):
                            sent_pos_type = "a"

                        elif p.startswith("V") or p.startswith("T"):
                            sent_pos_type = "v"

                        elif p.startswith("R"):
                            sent_pos_type = "r"

                        if sent_pos_type is not None:
                            sent_word = "%s/%s" % (sent_pos_type, t.lower())
                            if sent_word in self.sentiwordnet:
                                p_val, n_val = self.sentiwordnet[sent_word]
                                neutral_val = 1 - p_val - n_val
                                if p_val > n_val:
                                    if p_val > neutral_val:
                                        return "positive"
                                    else:
                                        return "neutral"
                                elif n_val > p_val:
                                    if n_val > neutral_val:
                                        return "negative"
                                    else:
                                        return "neutral"
                                elif n_val == p_val:
                                    return "neutral"
                            else:
                                return "none"


#LIWC Emotion Features

emotions ={
    'Posemo' : ["accept", "accepta", "accepted", "accepting", "accepts", "active", "admir", "ador", "advantag", "adventur", "affection", "agree", "agreeab", "agreed", "agreeing", "agreement", "agrees", "alright", "amaz", "amor", "amus", "aok", "appreciat", "assur", "attachment", "attract", "award", "awesome", "beaut", "beloved", "benefic", "benefit", "benefits", "benefitt", "benevolen", "benign", "best", "better", "bless", "bold", "bonus", "brave", "bright", "brillian", "calm", "care", "cared", "carefree", "careful", "cares", "caring", "casual", "casually", "certain", "challeng", "champ", "charit", "charm", "cheer", "cherish", "chuckl", "clever", "comed", "comfort", "commitment", "compassion", "compliment", "confidence", "confident", "confidently", "considerate", "contented", "contentment", "convinc", "cool", "courag", "create", "creati", "credit", "cute", "cutie", "daring", "darlin", "dear", "definite", "definitely", "delectabl", "delicate", "delicious", "deligh", "determina", "determined", "devot", "digni", "divin", "dynam", "eager", "ease", "easie", "easily", "easiness", "easing", "easy", "ecsta", "efficien", "elegan", "encourag", "energ", "engag", "enjoy", "entertain", "enthus", "excel", "excit", "fab", "fabulous", "faith", "fantastic", "favor", "favour", "fearless", "festiv", "fiesta", "fine", "flatter", "flawless", "flexib", "flirt", "fond", "fondly", "fondness", "forgave", "forgiv", "free", "free", "freeb", "freed", "freeing", "freely", "freeness", "freer", "frees", "friend", "fun", "funn", "genero", "gentle", "gentler", "gentlest", "gently", "giggl", "giver", "giving", "glad", "gladly", "glamor", "glamour", "glori", "glory", "good", "goodness", "gorgeous", "grace", "graced", "graceful", "graces", "graci", "grand", "grande", "gratef", "grati", "great", "grin", "grinn", "grins", "ha", "haha", "handsom", "happi", "happy", "harmless", "harmon", "heartfelt", "heartwarm", "heaven", "heh", "helper", "helpful", "helping", "helps", "hero", "hilarious", "hoho", "honest", "honor", "honour", "hope", "hoped", "hopeful", "hopefully", "hopefulness", "hopes", "hoping", "hug ", "hugg", "hugs", "humor", "humour", "hurra", "ideal", "importan", "impress", "improve", "improving", "incentive", "innocen", "inspir", "intell", "interest", "invigor", "joke", "joking", "joll", "joy", "keen", "kidding", "kind", "kindly", "kindn", "kiss", "laidback", "laugh", "libert", "like", "likeab", "liked", "likes", "liking", "livel", "LMAO", "LOL", "love", "loved", "lovely", "lover", "loves", "loving", "loyal", "luck", "lucked", "lucki", "lucks", "lucky", "madly", "magnific", "merit", "merr", "neat", "nice", "nurtur", "ok", "okay", "okays", "oks", "openminded", "openness", "opportun", "optimal", "optimi", "original", "outgoing", "painl", "palatabl", "paradise", "partie", "party", "passion", "peace", "perfect", "play", "played", "playful", "playing", "plays", "pleasant", "please", "pleasing", "pleasur", "popular", "positiv", "prais", "precious", "prettie", "pretty", "pride", "privileg", "prize", "profit", "promis", "proud", "radian", "readiness", "ready", "reassur", "relax", "relief", "reliev", "resolv", "respect ", "revigor", "reward", "rich", "ROFL", "romanc", "romantic", "safe", "satisf", "save", "scrumptious", "secur", "sentimental", "share", "shared", "shares", "sharing", "silli", "silly", "sincer", "smart", "smil", "sociab", "soulmate", "special", "splend", "strength", "strong", "succeed", "success", "sunnier", "sunniest", "sunny", "sunshin", "super", "superior", "support", "supported", "supporter", "supporting", "supportive", "supports", "suprem", "sure", "surpris", "sweet", "sweetheart", "sweetie", "sweetly", "sweetness", "sweets", "talent", "tehe", "tender", "terrific", "thank", "thanked", "thankf", "thanks", "thoughtful", "thrill", "toleran", "tranquil", "treasur", "treat", "triumph", "true ", "trueness", "truer", "truest", "truly", "trust", "truth", "useful", "valuabl", "value", "valued", "values", "valuing", "vigor", "vigour", "virtue", "virtuo", "vital", "warm", "wealth", "welcom", "well", "win", "winn", "wins", "wisdom", "wise", "won", "wonderf", "worship", "worthwhile", "wow", "yay", "yays"],

    'Negemo' : ["abandon", "abuse", "abusi", "ache", "aching", "advers", "afraid", "aggravat", "aggress", "agitat", "agoniz", "agony", "alarm", "alone", "anger", "angr", "anguish", "annoy", "antagoni", "anxi", "apath", "appall", "apprehens", "argh", "argu", "arrogan", "asham", "assault", "asshole", "attack", "aversi", "avoid", "awful", "awkward", "bad", "bashful", "bastard", "battl", "beaten", "bitch", "bitter", "blam", "bore", "boring", "bother", "broke", "brutal", "burden", "careless", "cheat", "complain", "confront", "confus", "contempt", "contradic", "crap", "crappy", "craz", "cried", "cries", "critical", "critici", "crude", "cruel", "crushed", "cry", "crying", "cunt", "cut", "cynic", "damag", "damn", "danger", "daze", "decay", "defeat", "defect", "defenc", "defens", "degrad", "depress", "depriv", "despair", "desperat", "despis", "destroy", "destruct", "devastat", "devil", "difficult", "disadvantage", "disagree", "disappoint", "disaster", "discomfort", "discourag", "disgust", "dishearten", "disillusion", "dislike", "disliked", "dislikes", "disliking", "dismay", "dissatisf", "distract", "distraught", "distress", "distrust", "disturb", "domina", "doom", "dork", "doubt", "dread", "dull", "dumb", "dump", "dwell", "egotis", "embarrass", "emotional", "empt", "enemie", "enemy", "enrag", "envie", "envious", "envy", "evil", "excruciat", "exhaust", "fail", "fake", "fatal", "fatigu", "fault", "fear", "feared", "fearful", "fearing", "fears", "feroc", "feud", "fiery", "fight", "fired", "flunk", "foe", "fool", "forbid", "fought", "frantic", "freak", "fright", "frustrat", "fuck", "fucked", "fucker", "fuckin", "fucks", "fume", "fuming", "furious", "fury", "geek", "gloom", "goddam", "gossip", "grave", "greed", "grief", "griev", "grim", "gross", "grouch", "grr", "guilt", "harass", "harm", "harmed", "harmful", "harming", "harms", "hate", "hated", "hateful", "hater", "hates", "hating", "hatred", "heartbreak", "heartbroke", "heartless", "hell", "hellish", "helpless", "hesita", "homesick", "hopeless", "horr", "hostil", "humiliat", "hurt", "idiot", "ignor", "immoral", "impatien", "impersonal", "impolite", "inadequa", "indecis", "ineffect", "inferior ", "inhib", "insecur", "insincer", "insult", "interrup", "intimidat", "irrational", "irrita", "isolat", "jaded", "jealous", "jerk", "jerked", "jerks", "kill", "lame", "lazie", "lazy", "liabilit", "liar", "lied", "lies", "lone", "longing", "lose", "loser", "loses", "losing", "loss", "lost", "lous", "low", "luckless", "ludicrous", "lying", "mad", "maddening", "madder", "maddest", "maniac", "masochis", "melanchol", "mess", "messy", "miser", "miss", "missed", "misses", "missing", "mistak", "mock", "mocked", "mocker", "mocking", "mocks", "molest", "mooch", "moodi", "moody", "moron", "mourn", "murder", "nag", "nast", "needy", "neglect", "nerd", "nervous", "neurotic", "numb", "obnoxious", "obsess", "offence", "offend", "offens", "outrag", "overwhelm", "pain", "pained", "painf", "paining", "pains", "panic", "paranoi", "pathetic", "peculiar", "perver", "pessimis", "petrif", "pettie", "petty", "phobi", "piss", "piti", "pity ", "poison", "prejudic", "pressur", "prick", "problem", "protest", "protested", "protesting", "puk", "punish", "rage", "raging", "rancid", "rape", "raping", "rapist", "rebel", "reek", "regret", "reject", "reluctan", "remorse", "repress", "resent", "resign", "restless", "revenge", "ridicul", "rigid", "risk", "rotten", "rude", "ruin", "sad", "sadde", "sadly", "sadness", "sarcas", "savage", "scare", "scaring", "scary", "sceptic", "scream", "screw", "selfish", "serious", "seriously", "seriousness", "severe", "shake", "shaki", "shaky", "shame", "shit", "shock", "shook", "shy", "sicken", "sin", "sinister", "sins", "skeptic", "slut", "smother", "smug", "snob", "sob", "sobbed", "sobbing", "sobs", "solemn", "sorrow", "sorry", "spite", "stammer", "stank", "startl", "steal", "stench", "stink", "strain", "strange", "stress", "struggl", "stubborn", "stunk", "stunned", "stuns", "stupid", "stutter", "submissive", "suck", "sucked", "sucker", "sucks", "sucky", "suffer", "suffered", "sufferer", "suffering", "suffers", "suspicio", "tantrum", "tears", "teas", "temper", "tempers", "tense", "tensing", "tension", "terribl", "terrified", "terrifies", "terrify", "terrifying", "terror", "thief", "thieve", "threat", "ticked", "timid", "tortur", "tough", "traged", "tragic ", "trauma", "trembl", "trick", "trite", "trivi", "troubl", "turmoil", "ugh", "ugl", "unattractive", "uncertain", "uncomfortabl", "uncontrol", "uneas", "unfortunate", "unfriendly", "ungrateful", "unhapp", "unimportant", "unimpress", "unkind", "unlov", "unpleasant", "unprotected", "unsavo", "unsuccessful", "unsure", "unwelcom", "upset", "uptight", "useless ", "vain", "vanity", "vicious", "victim", "vile", "villain", "violat", "violent", "vulnerab", "vulture", "war", "warfare", "warred", "warring", "wars", "weak", "weapon", "weep", "weird", "wept", "whine", "whining", "whore", "wicked", "wimp", "witch", "woe", "worr", "worse", "worst", "worthless ", "wrong", "yearn"]
}


class LIWCEmotionFeature(TfidfVectorizer):
    """
    Emotion feature estimator
    """
    #_sentiwordnet_resource = os.path.join(os.path.dirname(__file__), 'resources')

    # def __init__(self):
    #    self.sentiwordnet = load_sentiwordnet(os.path.join(self._sentiwordnet_resource,'SentiWordNet_3.0.0_20130122.txt'))

    def build_analyzer(self):
        analyzer = super(TfidfVectorizer,
                         self).build_analyzer()
        self.emotion_dict = emotions
        # lambda doc: (self._get_sentiments(w) for w in analyzer(doc.tagged_data))
        return lambda doc: (self._get_sentiments(w) for w in analyzer(doc.content))

    def _get_sentiments(self, w):
        # print("I am here")
        flag = False
        for word in self.emotion_dict["Posemo"]:
            if word in w:
                return w+"/positive"
                flag = True

        for word in self.emotion_dict["Negemo"]:
            if word in w:
                return w+"/negative"
                flag = True

        if flag == False:
            return w+"/none"





# Sentic Concepts Features


class SenticConceptsTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer,
                         self).build_analyzer()
        return lambda doc: (w for w in analyzer(doc.concepts))


class SenticConceptsScores(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(
            ['avg_sensitivity', 'avg_attention', 'avg_pleasantness', 'avg_aptitude', 'avg_polarity',
             #        'max_sensitivity',
             # 'max_attention', 'max_pleasantness', 'max_aptitude', 'max_polarity', 'min_sensitivity', 'min_attention',
             # 'min_pleasantness', 'min_aptitude', 'min_polarity'
             ])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        feature_vector = []
        for doc in documents:
            avg_sensitivity = np.mean(doc.sensitivity)
            # min_sensitivity = np.min(doc.sensitivity)
            # max_sensitivity = np.max(doc.sensitivity)

            avg_attention = np.mean(doc.attention)
            # min_attention = np.min(doc.attention)
            # max_attention = np.max(doc.attention)

            avg_pleasantness = np.mean(doc.pleasantness)
            # min_pleasantness = np.min(doc.pleasantness)
            # max_pleasantness = np.max(doc.pleasantness)

            avg_aptitude = np.mean(doc.aptitude)
            # min_aptitude = np.min(doc.aptitude)
            # max_aptitude = np.max(doc.aptitude)

            avg_polarity = np.mean(doc.polarity)
            # min_polarity = np.min(doc.polarity)
            # max_polarity = np.max(doc.polarity)

            feature_vector.append(
                [avg_sensitivity, avg_attention, avg_pleasantness, avg_aptitude, avg_polarity,
                 # max_sensitivity,
                 # max_attention, max_pleasantness, max_aptitude, max_aptitude, max_polarity, min_attention,
                 # min_pleasantness, min_aptitude, min_aptitude, min_polarity
                 ])

        return np.array(feature_vector)
