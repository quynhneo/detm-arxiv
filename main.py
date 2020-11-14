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
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *  # stemmer

nltk.download('wordnet')  # lexical database
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet

# load data
from subprocess import check_output

data = pd.read_csv('../data_undebates_kaggle/un-general-debates.csv')
print('data shape:', data.shape)
data.head()
data_text = data['text']

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
    for token in gensim.utils.simple_preprocess(document):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


#  Select a document to preview after preprocessing.
doc_sample = data_text[1]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')

print(preprocess(doc_sample))

list_of_doc = []  # list of doc, like sentences in gensim documentation
for doc in data_text[0:2]:
    list_of_doc.append(preprocess(doc))



EMB_DIM = 300

# data_text would work as input too, but to stick to the documentation
# x = data_text.values.tolist()  # list of all statements
# y = [[f] for f in x]  # list of list

# need to update with original settings
model = Word2Vec(list_of_doc, size=EMB_DIM, window=15, min_count=5, negative=15, iter=10,
                 workers=multiprocessing.cpu_count())
model.save('w2v.model')