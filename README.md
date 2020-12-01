# How topics of STEM papers change with time, using Dynamic Embedded Topic Modeling 
Running dynamic embedded topic modeling on abstracts of arxiv articles

## Get the meta data, containing abstracts: 
https://www.kaggle.com/Cornell-University/arxiv

## Generate embedding 
`python word2vec/run_w2v.py`
This will produce vector representation of words (default dimension = 300), and save as embed.txt

## Preprocess text data 
