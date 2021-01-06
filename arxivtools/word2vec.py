import multiprocessing
from itertools import repeat
import timeit


from gensim.models import Word2Vec
from preprocessing import read_meta_data, preprocess, frequency_filter

start_time = timeit.default_timer()
from typing import List

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


if __name__ == '__main__':
    # Read stopwords
    with open('stops.txt', 'r') as f:
        stops = f.read().split('\n')

    # read corpus
    meta_data_file = '../../arxiv-metadata-oai-snapshot.json'
    all_docs_ini,_ = read_meta_data(meta_data_file, 'hep-ph')  # read all abstracts in hep-ph

    print('preprocessing')
    print('number of cpus: ', multiprocessing.cpu_count())
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    list_of_list = pool.starmap(preprocess, zip(all_docs_ini, repeat(stops)))

    pool.close()
    pool.join()
    del all_docs_ini

    list_of_list_filtered = frequency_filter(list_of_list, 30, 0.7, multiprocessing.cpu_count())

    print('break')
    del list_of_list

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
