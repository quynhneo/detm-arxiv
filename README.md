# Time evolution of STEM topics with Dynamic Embedded Topic Modeling (work in progress)
Quynh M. Nguyen<sup> a, b</sup> and Kyle Cranmer<sup> a, c</sup>

<sup> a</sup> _Physics Department, New York University, New York 10003_

<sup> b</sup> _Applied Math Lab, Courant Institute, New York University, New York 10012_

<sup> c</sup> _Center for Data Science, New York University, New York 10011_

## Project description
Running dynamic embedded topic modeling on abstracts of arxiv articles and discover how topics in STEM change in time. This is an implementation of [Dynamic Embedded Topic Modeling](https://github.com/adjidieng/DETM) by Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei of Columbia University. 

## Get the abstracts 
Visit https://www.kaggle.com/Cornell-University/arxiv to get `arxiv-metadata-oai-snapshot.json` which contain about 2 millions records, each has a dozen of fields but we are interested in `abstract`, `categories`, and `update_date`.

## Generate embedding 
Modify  path to  `arxiv-metadata-oai-snapshot.json` in `word2vec/run_w2v.py` and run:
  
```
python word2vec/run_w2v.py
```

This will: read in abstracts, remove punctuations, remove stop words listed in [`word2vec/stops.txt`](https://github.com/quynhneo/detm-arxiv/blob/master/word2vec/stops.txt), remove words that appear in less than 30 abstracts, and words appear in more than 70% of abstracts, and produces vector representations of all the words (default embedding dimension = 300) using original settings from [Mikolov 2013 NIPS paper](https://arxiv.org/pdf/1310.4546.pdf), and save as embeddings.txt. The process takes about an hour per 150,000 abstracts on a laptop. 

## Clone my fork of the original [Dynamic Embedded Topic Modeling](https://github.com/adjidieng/DETM)
I have made some changes to because of runtime errors, no change to the model so far
```
git clone https://github.com/quynhneo/DETM
```
The environtment could be set up by pip or conda, for example, using conda:
```
conda create --name detm --file requirements.txt 
conda activate detm
```

## Preprocess text data 
Modify  path to  `arxiv-metadata-oai-snapshot.json` in `scripts/data_undebates.py` and run:
```
python scripts/data_undebates.py
```
This will take about 5 minutes per 150,000 abstracts on a laptop. The output (`.mat` files) will be save in `script/split_paragraph_False/`
## Run Dynamic Embedded Topic Modeling 
Modify paths to preprocess text data, and `embeddings.txt`, and other models settings are on top of the file, and run:

```
python main.py
``` 
 This stage will take much longer and should be run with GPU (CPU mode is too slow even with a 16 cores)

More instruction for running on a cluster using CUDA is [here](https://github.com/quynhneo/detm-arxiv/blob/master/docs/singularity_slurm.md)

Output will be 3 `.mat` files in `results`. 
## Plot the results
Edit `beta_file` in `plot_word_evolution.py` to be the path to the file ending in `_beta` in `results` and run:
```
python plot_word_evolution.py 
```


**A very preliminary** result, evolution of word probability across time for eight different topics is shown in the [.png file](https://github.com/quynhneo/detm-arxiv/blob/master/detm_un_K_50_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_beta.png). A lot more pruning and tuning to be done. Currently, the run time is too long, and the text has to be preprocessed more (for example the group of words including  'abstract','introduction','conclusion','method'... is learned as a topic because they always appear together)  
