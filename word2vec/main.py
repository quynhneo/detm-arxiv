import nltk
import gensim
from gensim.models import Word2Vec
import multiprocessing
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS, preprocess_string, preprocess_documents, strip_punctuation, \
    strip_numeric
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *  # stemmer
from nltk.corpus import wordnet

nltk.download('wordnet')  # lexical database
nltk.download('averaged_perceptron_tagger')


def get_wordnet_pos(tok):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([tok])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_stemming(token):
    """ lemmatize the token """
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(token, pos=get_wordnet_pos(token)))


def preprocess(document):
    """ tokenization, remove stopwords and words with less than 3 characters
     INPUT: a document
     OUTPUT: a list of token"""
    result = []

    for token in gensim.utils.simple_preprocess(document, min_len=3):
        # This lower cases, tokenizes, de-accents (optional). also remove number, punctuation, and some stop words
        # the output are final tokens = unicode strings
        if token not in gensim.parsing.preprocessing.STOPWORDS:  # remove stop words from a list
            #  result.append(lemmatize_stemming(token))
            result.append(token)
    return result

# word appears more than 70% of document?
# word appears in less than 30 document? (un debate)
# parallel

# load data
from subprocess import check_output

data = pd.read_csv('../data_undebates_kaggle/un-general-debates.csv')
print('data shape:', data.shape)
data.head()
data_text = data['text']
#
# #  Select a document to preview after preprocessing.
#
# doc_sample = data_text[2]
#
# print('original document: ')
# words = []
# for word in doc_sample.split(' '):
#     words.append(word)
# print(words)
#
# print('processed: \n', preprocess(doc_sample))
#
# list_of_doc = data_text.map(preprocess)  # list of doc, like 'sentences' in gensim documentation
# #  get a dictionary: key: integer id, value: word (str)
# dictionary = gensim.corpora.Dictionary(list_of_doc)
# dictionary.filter_extremes(no_below=30, no_above=0.7)  # list of none-extreme words Dieng's setting
# #  apply filter to the list of doc
# list_of_doc_filtered = [[word for word in doc if word in dictionary.token2id] for doc in list_of_doc]
#
# #
# #
#
# # list_of_doc = []  # list of doc, like 'sentences' in gensim documentation
# # for doc in data_text:
# #     list_of_doc.append(preprocess(doc))
# #
# EMB_DIM = 300  # embedding dimension
#
# # need to compare with Dieng's settings
# model = Word2Vec(list_of_doc_filtered, size=EMB_DIM, window=15, min_count=5, negative=15, iter=10,
#                  workers=multiprocessing.cpu_count(), sg=1, hs=1, sample=0.00001)
# #  setting from original word2vec paper
# #  min_count (int, optional) – Ignores all words with total frequency lower than this.
# #  sg skip gram
# #  hs hierarchical softmax
# #  sample subsampling threshold (high frequency are sampled less)
#
# model.save('w2v.model')
# w2v_out = model.wv
# w2v_out.vectors.shape # number of words x embedded dimension
# w2v_out.vocab
#
# with open("embed.txt","w") as outputfile:
#     for word, vec in zip(w2v_out.vocab, w2v_out.vectors):
#         print(word, *vec, sep=" ",file=outputfile )
