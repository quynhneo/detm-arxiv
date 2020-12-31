import sys
import multiprocessing
from itertools import repeat
import nltk
import gensim
from gensim.models import Word2Vec
from typing import List, Dict
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
import json
import string
import timeit

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS, preprocess_string, preprocess_documents, strip_punctuation, \
    strip_numeric
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *  # stemmer
from nltk.corpus import wordnet
nltk.download('wordnet')  # lexical database
nltk.download('averaged_perceptron_tagger')

start_time = timeit.default_timer()

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


def preprocess_(document):
    """ tokenization, remove stopwords and words with less than 3 characters
     INPUT: a document
     OUTPUT: a list of token"""
    result = []

    for token in gensim.utils.simple_preprocess(document, min_len=2):
        # This lower cases, de-accents (optional), filter words shorter than 2 chars
        # and TOKENIZE: remove number
        # the output are final tokens = unicode strings
        if token not in gensim.parsing.preprocessing.STOPWORDS:  # remove stop words from a list
            #  result.append(lemmatize_stemming(token))
            result.append(token)
    return result


def preprocess(document, stopwords):
    """ tokenize, lower case,
    remove punctuation: '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~', and new line character
    latex expressions are not processed
    hyphens (e.g. kaluza-klein), and numbers (e.g. 3D), and accent are allowed
    remove stopwords and words with less than 1 characters
    INPUT: a string
    OUTPUT: a list of token"""
    result = []

    lemma = WordNetLemmatizer()

    for token in document.split():
        token = token.lower().replace("’", "").replace("'", "").replace("\n", " ").translate(
            str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))
        #token = lemma.lemmatize(token)
        if len(token) > 1 and token.islower() and token not in stopwords:
            result.append(token)

    return result


def read_data(json_file, category=None):
    # Read raw data
    print('reading raw data...')

    file = open(json_file, 'r')
    line_count = 0
    all_docs = []
    for line in file:  # 1.7m
        # if line_count > 1000: # uncomment this to do quick test run
        #     break
        try:
            # line_view = json.loads(file.readline())  # view object of the json line
            line_view = json.loads(line)  # view object of the json line
            # print(line_view['update_date'][0:4])
            # print(line_view['abstract'])
        except:
            print('bad line', line_count) # 896728
            print(line_view)
            line_count += 1
            continue
        if category is None:
            all_docs.append(line_view['abstract'])  # list of document strings,  ["it is indeed ...", ...]
            line_count += 1
        elif category in line_view['categories']:  # select only 1 category
            all_docs.append(line_view['abstract'])  # list of document strings,  ["it is indeed ...", ...]
            line_count += 1
            # if line_count >10:
            #    break

    print("number of line is : ", line_count)
    file.close()
    return all_docs

# ------for reading from csv file ------

# data = pd.read_csv('../../data_undebates_kaggle/un-general-debates.csv')
# print('data shape:', data.shape)
# data.head()
# data_text = data['text']
# print(type (data_text))
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


def rm_extremes(doc: List[str], to_keep_dict: List[str]) -> List[str]:
    return [w for w in doc if w in to_keep_dict]


if __name__ == '__main__':
    # Read stopwords
    with open('stops.txt', 'r') as f:
        stops = f.read().split('\n')

    # read corpus
    meta_data_file = '../../arxiv-metadata-oai-snapshot.json'
    all_docs_ini = read_data(meta_data_file, 'hep-ph')  # read all abstracts in hep-ph

    print('preprocessing')
    print('number of cpus: ', multiprocessing.cpu_count())
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    list_of_list = pool.starmap(preprocess, zip(all_docs_ini, repeat(stops)))


    del all_docs_ini

    #  get a dictionary: key: integer id, value: word (str)
    print('getting dictionary')
    dictionary = gensim.corpora.Dictionary(list_of_list)
    # dictionary encapsulates the mapping between normalized words and their integer ids.

    print('filter out extremes frequencies')
    dictionary.filter_extremes(no_below=30, no_above=0.7)
    #  dictionary of none-extreme words -- Dieng & Blei 2019 setting
    #  apply filter to the list of doc
    #  no_below: minimum document frequency (int)
    #  no_above: maximum document frequency (float [0,1])

    print('apply filters')
    # for each document in the list of document, select only words in the dictionary, and not in list of stopwords
    # list_of_list_filtered = [[word for word in doc if word in dictionary.token2id] for doc in list_of_list]
    list_of_list_filtered = pool.starmap(rm_extremes, zip(list_of_list, repeat(dictionary.token2id)))

    print('break')

    del dictionary
    del list_of_list
    # pool.close()
    # pool.join()

    EMB_DIM = 300  # embedding dimension

    # need to compare with Dieng & Blei 2019 settings
    print('running word2vec')
    model = Word2Vec(list_of_list_filtered, size=EMB_DIM, window=15, min_count=5, negative=15, iter=10,
                     workers=multiprocessing.cpu_count(), sg=1, hs=1, sample=0.00001)
    #  settings from original word2vec paper (Mikolov 2013 NIPS)
    #  min_count (int, optional): Ignores all words with total frequency lower than this.
    #  sg: skip gram (vs bow)
    #  hs: hierarchical softmax
    #  sample: subsampling threshold (high frequency are sampled less)

    model.save('w2v.model')
    w2v_out = model.wv
    # w2v_out.vectors.shape  # number of words x embedded dimension
    # w2v_out.vocab

    with open("embeddings.txt", "w") as out_put_file:
        for word, vec in zip(w2v_out.vocab, w2v_out.vectors):
            print(word, *vec, sep=" ", file=out_put_file)

stop_time = timeit.default_timer()
print("run time {}".format((stop_time-start_time)/3600))
