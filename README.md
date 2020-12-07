 _THIS REPO IS HERE TEMPORARILY, WILL BE MOVED TO MY PERSONAL [GITHUB](https://github.com/quynhneo)_ )

# Time evolution of STEM topics with Dynamic Embedded Topic Modeling
Quynh M. Nguyen<sup> a</sup> and Kyle Cranmer<sup> a, b</sup>

<sup> a</sup> _Physics Department, New York University, New York 10012_

<sup> a, b</sup> _Center for Data Science, New York University, New York 10011_

## Project description
Running dynamic embedded topic modeling on abstracts of arxiv articles and discover how topics in STEM change in time. This is an implementation of [Dynamic Embedded Topic Modeling](https://github.com/adjidieng/DETM) by Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei of Columbia University. 

## Get the meta data, containing abstracts 
Visit https://www.kaggle.com/Cornell-University/arxiv (json format)

## Generate embedding 
  
```
python word2vec/run_w2v.py
```
  
This will take all words from abstracts, apply nlp processing (remove stop words, remove rare words, etc) and produce vector representations of all the words (default embedding dimension = 300), and save as embed.txt.

More on word embedding is available in this paper: https://arxiv.org/pdf/1310.4546.pdf
## Clone my fork of the original [Dynamic Embedded Topic Modeling](https://github.com/adjidieng/DETM)
I have made some changes to becasuse runtime errors, no change to the model
```
git clone https://github.com/quynhneo/DETM_arxiv_org
```

## Preprocess text data 
```
python data_undebates.py
```
https://github.com/quynhneo/DETM_arxiv_org/blob/master/scripts/data_undebates.py (modify the path to json file appropriately)

## Run Dynamic Embedded Topic Modeling 

```
python main.py
``` 
in https://github.com/quynhneo/DETM_arxiv_org/blob/master/main.py (all setup and models settings are on top of the file).
