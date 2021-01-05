import json
from typing import List

import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
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


def preprocess(document: str, stopwords: List[str]) -> List[str]:
    """
    INPUT: a string
    OUTPUT: a list of token

        tokenize, lower case,
        remove punctuation: '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~', and new line character
        remove stop words from provided list and words with less than 1 characters from document
    note:
        latex expressions are not processed
        hyphens (e.g. kaluza-klein), and numbers (e.g. 3D), and accent are allowed
    """
    result = []

    lemma = WordNetLemmatizer()

    for token in document.split():
        token = token.lower().replace("’", "").replace("'", "").replace("\n", " ").translate(
            str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))  # modified from python string.punctuation

        if len(token) > 1 and token.islower() and token not in stopwords:
            token = lemma.lemmatize(token, pos=get_wordnet_pos(token))  # plural-> singular, Verb-ing to verb, etc
            # doesn't work for all words
            result.append(token)

    return result


def read_meta_data(json_file: json, category: str = None) -> List[str]:
    """
    return a list of abstracts from the meta data file
    category: include all categories if not specified

    """
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


def rm_extremes(doc: List[str], to_keep_dict: List[str]) -> List[str]:
    """remove str element in doc if it's not also in to_keep_dict"""
    return [w for w in doc if w in to_keep_dict]
