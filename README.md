# Time evolution of STEM topics with Dynamic Embedded Topic Modeling (work in progress)
Quynh M. Nguyen<sup> a, b</sup> and Kyle Cranmer<sup> a, c</sup>

<sup> a</sup> _Physics Department, New York University, New York 10003_

<sup> b</sup> _Applied Math Lab, Courant Institute, New York University, New York 10012_

<sup> c</sup> _Center for Data Science, New York University, New York 10011_

## Project description
Running dynamic embedded topic modeling on abstracts of arxiv articles and discover how topics in STEM change in time. This is an implementation of [Dynamic Embedded Topic Modeling](https://github.com/adjidieng/DETM) by Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei of Columbia University. 

## Get the abstracts 
Visit https://www.kaggle.com/Cornell-University/arxiv to get `arxiv-metadata-oai-snapshot.json` which contains about 2 million records, each has a dozen of fields, and we are interested in `abstract`, `categories`, and `update_date`.

## Generate embedding with `word2vec` 
Modify the path to `arxiv-metadata-oai-snapshot.json` in `word2vec/run_w2v.py` and run:
  
```
python word2vec/run_w2v.py
```

This will read in abstracts, remove punctuations, remove stop words listed in [`word2vec/stops.txt`](https://github.com/quynhneo/detm-arxiv/blob/master/word2vec/stops.txt), remove rare words that appear in less than 30 abstracts, and words appear in more than 70% of abstracts, and produces vector representations of all the words left (default embedding dimension = 300) using original settings from [Mikolov 2013 NIPS paper](https://arxiv.org/pdf/1310.4546.pdf). The ressults are save as `embeddings.txt` where each line is a word following by 300 numbers. The process takes about an hour per 150,000 abstracts on a laptop. 

## Clone our fork of the original repo
We have made some changes to fix runtime errors, no change to the model in this fork:
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
 This stage will take much longer and should be run with GPU (CPU mode is too slow even with a 16 cores node)

More instruction for running on a cluster using CUDA is [here](https://github.com/quynhneo/detm-arxiv/blob/master/docs/singularity_slurm.md)

Output will be 3 `.mat` files in `results`. 
## Plot the results
Edit `beta_file` in `plot_word_evolution.py` to be the path to the file ending in `_beta` in `results` and run:
```
python plot_word_evolution.py 
```


## Preliminary results
The plot below shows results for DETM on [`hep-ph`](https://arxiv.org/archive/hep-ph) (high energy physics phenomenology) category. Assuming there are 50 topics, the 6 most meaningful ones were  manually. For each topics, probabilities of some top words are plotted against time (2007-2020). 
For example, topic 46 shows the increase in `higgs` coinciding with the discovery of Higgs boson in 2012.

![result](https://github.com/quynhneo/detm-arxiv/blob/master/detm_un_K_50_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_0.001_Bsz_200_RhoSize_300_L_4_minDF_30_trainEmbeddings_1_beta.png)

Tracking the word `diphoton`, its peak probability coincides with the [flurry of papers](https://en.wikipedia.org/wiki/750_GeV_diphoton_excess) on a possible discovery of new physics around 2015-2016, which turned out to be just a statistical fluke. 

![diphoton](https://github.com/quynhneo/detm-arxiv/blob/master/diphoton.png)
(`diphoton` topic contribution is dominated by other words in topic 3 and 46 which are not shown here)

The above plots are from running 400 epoches on data of 150,000 abstracts of `hep-ph`. We use 4 Nvidia RTX8000 GPUs and the runtime was 13 hours.
