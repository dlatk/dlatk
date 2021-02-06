# Differential Language Analysis ToolKit

DLATK is an end to end human text analysis package for Python 3. It is specifically *suited for social media, Psychology, and health research*, developed originally for projects out of the University of Pennsylvania and Stony Brook University.  Currently, it has been used in over 75 peer-reviewed publicaitons (many from before there was an article to reference). 

DLATK can perform:

- linguistic feature extraction (i.e. turning text into variables)
- differential language analysis (i.e. finding the language that is most associated with psychological or health variables)
- wordcloud visualization
- statistical- and machine learning-based supervised prediction (regresion and classification)
- statistical- and machine learning-based dimensionality reduction and clustering
- mediation analysis
- contextual embeddings: using deep learning transformers message, user, or group embeddings
- part-of-speech tagging

DLATK can integrate with
- [pandas](http://pandas.pydata.org/) dataframe output

Functions of DLATK use:

- [HuggingFace](http://??.org/) for transformer language models
- [Mallet](http://mallet.cs.umass.edu/) for creating LDA topics
- [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml) for syntactic parsing
- [CMU's TweetNLP](http://www.cs.cmu.edu/~ark/TweetNLP/) for POS tagging; alternative tokenizing


## Installation

DLATK is available via any of four popular installation platforms: conda, pip, github, or Docker:

#### New to installing Python packages?
It is recommended that you see the [full installation instructions](http://dlatk.wwbp.org/install.html#dependencies). 

### 0. Make sure you have python3-mysqldb:
```sh
sudo apt-get install python3-mysqldb
```

### 1. conda
```sh
conda install -c wwbp dlatk
```

### 2. pip
```sh
pip install dlatk
```

### 3. GitHub
```sh
git clone https://github.com/dlatk/dlatk.git
cd dlatk
python setup.py install
```

### 4. Docker
Detailed Docker install instructions [here](http://dlatk.wwbp.org/tutorials/tut_docker.html).

```sh
docker run --name mysql_v5  --env MYSQL_ROOT_PASSWORD=my-secret-pw --detach mysql:5.5
docker run -it --rm --name dlatk_docker --link mysql_v5:mysql dlatk/dlatk bash
```

- [DLATK at DockerHub](https://hub.docker.com/r/dlatk/dlatk/)
- [DockerFile on GitHub](https://github.com/dlatk/dlatk-docker)

## Dependencies
- [mysqlclient](https://github.com/PyMySQL/mysqlclient-python)
- [NumPy](http://www.numpy.org)
- [scikit-learn](http://www.scikit-learn.org/)
- [SciPy](http://www.scipy.org/)
- [statsmodels](http://www.statsmodels.org/)

See the [full installation instructions](http://dlatk.wwbp.org/install.html#dependencies)
for recommended and optional dependencies.

## Documentation

The documentation for the latest release is at [dlatk.wwbp.org](dlatk.wwbp.org).

## Citation

If you use DLATK in your work please cite the following [paper](http://aclweb.org/anthology/D17-2010):

```
@InProceedings{DLATKemnlp2017,
  author =  "Schwartz, H. Andrew
    and Giorgi, Salvatore
    and Sap, Maarten
    and Crutchley, Patrick
    and Eichstaedt, Johannes
    and Ungar, Lyle",
  title =   "DLATK: Differential Language Analysis ToolKit",
  booktitle =   "Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
  year =  "2017",
  publisher =   "Association for Computational Linguistics",
  pages =   "55--60",
  location =  "Copenhagen, Denmark",
  url =   "http://aclweb.org/anthology/D17-2010"
}

```

## License

Licensed under a [GNU General Public License v3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Background

Developed by the [World Well-Being Project](http://www.wwbp.org) based out of the University of Pennsylvania and Stony Brook University.
