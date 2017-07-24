# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import VectorizerMixin
import numpy as np
import gensim
import nltk
from nltk.stem.snowball import SnowballStemmer
from preprocess import ark_tweet_tokenizer
import nltk
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
import string


__all__ = ['LDAFeatures']



class LDAFeatures(BaseEstimator, VectorizerMixin, TransformerMixin):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=False, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 analyzer='word', dtype=np.float32, model_name="/home/niloofar/Shahryar_Niloofar/all/niloofar/Niloofar/Topic_Modeling_results/wiki/num_topic=20/wiki_lda_uni.model"):
        #/home/niloofar/Shahryar_Niloofar/all/niloofar/Niloofar/Topic_Modeling_results/wiki/num_topic=20/wiki_lda_uni.model
        # model_name = "/home/shahryar/niloofar/Niloofar/ask_lda_uni_new.model"
        #/home/niloofar/Shahryar_Niloofar/all/niloofar/Niloofar/Topic_Modeling_results/ask/num_topic=50/ask_lda_uni.model
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = ark_tweet_tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.dtype = dtype
        self.model_name = model_name
        # TODO: add the max and min document frequency

    def get_feature_names(self):
        return np.array(['topic_' + str(i) for i in range(self.num_features_)])

    def _words(self, stop_words=None):
        """Turn tokens into a sequence of words after stop words filtering"""
        # handle stop words


        #add punctuations and combine stopwords from gensim, nltk
        stop_words= stopwords.words('english')+list(string.punctuation)+[stop for stop in STOPWORDS] +['u','ur','r',"'m",'``',"'s","'ll","'re","n't","'d","im","dont","youre","gonna", "shes",
                                                                                                       "wanna","aint","didnt","thats","lets","ok","bc","bye","yes","no","oh","youll","wouldnt",
                                                                                                       "hes","hell","she's","he's","don't","you're","i'm","it's","who's", "theyre", "youd", "doesnt",
                                                                                                       "dont", "were", "werent", "was", "wasnt", "okay", "k", "ok", "thanks", "thank", "so", "soo",
                                                                                                       "thankyou", "ive", "1", "2", "3", "4", "5", "6", "7", "8", "9", "youve","arent", "isnt", "gotta"]
        stop_words=set(stop_words)

        return stop_words

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        if callable(self.analyzer):
            return self.analyzer

        preprocess = self.build_preprocessor()

        if self.analyzer == 'word':
            stop_words = self.get_stop_words()
            tokenize = self.build_tokenizer()

            return lambda doc: self._words(
                tokenize(preprocess(self.decode(doc))), stop_words)

        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             self.analyzer)

    def make_feature_vec(self, doc, label):

        # Function to average all of the word vectors in a given
        # paragraph
        #
        # Pre-initialize an empty numpy array (for speed)
        #print("hereeeeeeeeeeeeeeeeeee")
        feature_vec = np.zeros((self.num_features_,), dtype=self.dtype)
        #
        topics_vec = []
        dic = {}
        prob = []
        for n, p in self.model_.get_document_topics(doc):
            #topics_vec.append(n)
            # prob.append(p)
            feature_vec[n] = p
            # dic[p] = n

        # prob.sort(key=float, reverse=True)
        # print(prob)
        # topics_vec.append(dic[prob[0]])
        # topics_vec.append(dic[prob[1]])
        # topics_vec.append(dic[prob[2]])
        # print("The label is: {}".format(label))
        # print("The topics are: {}".format(topics_vec))


        # Loop over each word in the review and, if it is in the model's
        # vocaublary, add its features vector to the total
        '''for word in words:
            if word in self.index2word_set_:
                nwords = nwords + 1.
                feature_vec = np.add(feature_vec, self.model_[word])'''
        #
        # Divide the result by the number of words to get the average
        '''if nwords>0.0:
            feature_vec = np.divide(feature_vec, nwords)'''
        #print(feature_vec)
        return feature_vec

    def fit(self, documents, y=None):
        # TODO: implement word2vec training
        if self.model_name:
            print( "Loading LDA")
            self.stopwords = self._words()
            self.model_ = gensim.models.LdaModel.load(self.model_name)
            # self.id2word_ = gensim.corpora.Dictionary.load('/home/shahryar/niloofar/Niloofar/ask.dict')
            self.id2word_ = gensim.corpora.Dictionary.load('/home/niloofar/Shahryar_Niloofar/all/niloofar/Niloofar/Topic_Modeling_results/wiki/num_topic=20/wiki.dict')
            #/home/niloofar/Shahryar_Niloofar/all/niloofar/Niloofar/Topic_Modeling_results/wiki/num_topic=20/wiki.dict
            #/home/niloofar/Shahryar_Niloofar/all/niloofar/Niloofar/Topic_Modeling_results/ask/num_topic=50/ask.dict
            self.num_features_ = 20
            '''doc = []
            for d in documents:
                doc.append(d.content)'''
            document = []
            self.stemmer_ = SnowballStemmer("english")
            '''for docu in doc:
                texts=[]
                for sent in nltk.sent_tokenize(docu):
                    tokens = [self.stemmer_.stem(t) for t in ark_tweet_tokenizer(sent) if t not in self.stopwords]
                    texts.extend(tokens)
                document.append(texts)'''
            #self.id2word_ = gensim.corpora.Dictionary(document)
            #self.doc2bow_ = [self.id2word_.doc2bow(text) for text in document]
            #self.tfidf_model = gensim.models.TfidfModel(self.doc2bow_, id2word=self.id2word_,normalize=True)
            #self.corpus_tfidf = self.tfidf_model[self.doc2bow_]
            #self.corpus_lda=self.model_[self.corpus_tfidf]
            # Index2word is a list that contains the names of the words in
            # the model's vocabulary. Convert it to a set, for speed
            #self.index2word_set_ = set(self.model_.index2word)
            print ("Done Loading vectors")
        else:
            # TODO: implement word2vec training
            '''print("Building Word2Vec")
            sentences = []
            for doc in documents:
                sentences.append(ark_tweet_tokenizer(doc.content))
            self.model_ = word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
            self.num_features_ = self.model_.syn0.shape[1]
            self.index2word_set_ = set(self.model_.index2word)
            print ("Done Loading vectors")'''
            pass


        return self

    def transform(self, documents):

        analyze = self.build_analyzer()

        # Preallocate a 2D numpy array, for speed
        doc_feature_vecs = np.zeros((len(documents), self.num_features_), dtype=self.dtype)
        #
        print(doc_feature_vecs.shape[1])

        # Loop through the reviews
        for i, doc in enumerate(documents):

            texts=[]
            for sent in nltk.sent_tokenize(doc.content):
                tokens = [t for t in ark_tweet_tokenizer(sent) if t not in self.stopwords]

                texts.extend(tokens)



            # Print a status message every 1000th review
            '''if i % 1000. == 0.:
                print( "Document %d of %d" % (i, len(documents)))'''
            #
            # Call the function (defined above) that makes average features vectors

            doc_feature_vecs[i] = self.make_feature_vec(self.model_[self.id2word_.doc2bow(texts)], doc.label)
            #print(i)


        return doc_feature_vecs
