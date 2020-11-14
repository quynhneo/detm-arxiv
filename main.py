import nltk
from gensim.models import Word2Vec
import multiprocessing
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator

# load data
from subprocess import check_output
data = pd.read_csv('../data_undebates_kaggle/un-general-debates.csv')
print('data shape:', data.shape)
data.head()

print('hello')