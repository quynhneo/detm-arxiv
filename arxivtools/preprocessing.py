import json
from typing import List
from itertools import repeat
import multiprocessing

import nltk
from nltk.stem import WordNetLemmatizer
import gensim

from nltk.corpus import wordnet
nltk.download('wordnet')  # lexical database


def get_wordnet_pos(tok):
    """Map Part Of Speech tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([tok])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# def lemmatize_stemming(token):
#     """ lemmatize the token """
#     stemmer = SnowballStemmer("english")
#     return stemmer.stem(WordNetLemmatizer().lemmatize(token, pos=get_wordnet_pos(token)))


# def preprocess_(document):
#     """ tokenization, remove stopwords and words with less than 3 characters
#      INPUT: a document
#      OUTPUT: a list of token"""
#     result = []
#
#     for token in gensim.utils.simple_preprocess(document, min_len=2):
#         # This lower cases, de-accents (optional), filter words shorter than 2 chars
#         # and TOKENIZE: remove number
#         # the output are final tokens = unicode strings
#         if token not in gensim.parsing.preprocessing.STOPWORDS:  # remove stop words from a list
#             #  result.append(lemmatize_stemming(token))
#             result.append(token)
#     return result


def preprocess(document: str, stopwords: List[str]) -> List[str]:
    """
    INPUT: a string
    OUTPUT: a list of token

        tokenize, lower case,
        remove punctuation: '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~', and new line character
        remove stop words from provided list and words with less than 1 characters from document
    note:
        latex expressions are not processed
        hyphens (e.g. kaluza-klein), and numbers (e.g. 3D, 750), and accent are allowed
    """
    result = []

    lemma = WordNetLemmatizer()
    punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'  # modified from python string.punctuation, removed hyphen

    # replace punctuation with white space
    document = document.lower().replace("’", " ").replace("'", " ").replace("\n", " ").translate(
        str.maketrans(punctuation, " "*len(punctuation)))

    for token in document.split():
        if len(token) > 1 and token not in stopwords:   # and token.islower() removes pure numeric
            token = lemma.lemmatize(token, pos=get_wordnet_pos(token))  # plural-> singular, Verb-ing to verb, etc
            # doesn't work for all words
            result.append(token)

    return result


def read_meta_data(json_file: json, category: str = None) -> (List[str], List[str]):
    """
    return
    a list of abstracts (str) from the meta data file
    a list of year in str
    category: include all categories if not specified

    """

    file = open(json_file, 'r')
    line_count = 0
    all_timestamps = []
    all_docs = []
    for line in file:  # 1.7m
        #if line_count > 5000: # uncomment this to do quick test run
        #    break
        try:
            # line_view = json.loads(file.readline())  # view object of the json line
            line_view = json.loads(line)  # view object of the json line
            # print(line_view['update_date'][0:4])
            # print(line_view['abstract'])
        except:
            print('bad line', line_count)  # 896728
            print(line_view)
            line_count += 1
            continue
        if category is None:
            all_timestamps.append(line_view['update_date'][0:4])  # get the year only in yyyy-mm-dd format
            # return list of year string ['1989','1989',...]
            all_docs.append(line_view['abstract'])  # list of document strings,  ["it is indeed ...", ...]
            line_count += 1
        elif category in line_view['categories']:  # select only 1 category
            all_timestamps.append(line_view['update_date'][0:4])  # get the year only in yyyy-mm-dd format
            all_docs.append(line_view['abstract'])  # list of document strings,  ["it is indeed ...", ...]
            line_count += 1
            # if line_count >10:
            #    break

    print("number of line is : ", line_count)
    file.close()
    return all_docs, all_timestamps


def rm_unlisted_words(doc: List[str], whitelist: List[str]) -> List[str]:
    """remove str element in doc if it's not also in whitelist"""
    return [w for w in doc if w in whitelist]


def frequency_filter(list_o_list: List[List[str]], min_docs: int,
                     max_portion: float, num_workers=multiprocessing.cpu_count()) -> List[List[str]]:
    """filter out words that appear in less than min_docs of document and more than max_portion of documents"""
    #  get a dictionary: key: integer id, value: word (str)
    print('getting dictionary')
    dictionary = gensim.corpora.Dictionary(list_o_list)
    # dictionary encapsulates the mapping between normalized words and their integer ids.

    print('filter out extremes frequencies')
    dictionary.filter_extremes(no_below=min_docs, no_above=max_portion)
    #  dictionary of none-extreme words -- Dieng & Blei 2019 setting
    #  apply filter to the list of doc
    #  no_below: minimum document frequency (int)
    #  no_above: maximum document frequency (float [0,1])

    print('number of cores: ', multiprocessing.cpu_count())
    pool_ = multiprocessing.Pool(processes=num_workers)

    print('apply filters')
    # for each document in the list of document, select only words in the dictionary, and not in list of stopwords
    list_o_list_filtered = pool_.starmap(rm_unlisted_words, zip(list_o_list, repeat(dictionary.token2id)))

    pool_.close()
    pool_.join()

    return list_o_list_filtered
