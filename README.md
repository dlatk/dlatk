# Differential Language Analysis ToolKit

DLATK is an end to end human text analysis package, specifically suited for social media and social scientific applications. It is written in Python 3 and developed by the World Well-Being Project at the University of Pennsylvania and Stony Brook University. 

It contains:

- feature extraction
- part-of-speech tagging
- correlation
- prediction and classification
- mediation 
- dimensionality reduction and clustering
- wordcloud visualization

DLATK can utilize:

- [Mallet](http://mallet.cs.umass.edu/) for creating LDA topics
- [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml) 
- [CMU's TweetNLP](http://www.cs.cmu.edu/~ark/TweetNLP/) 
- [pandas](http://pandas.pydata.org/) dataframe output

## Installation

DLATK is available via any of four popular installation platforms: conda, pip, github, or Docker:

#### New to installing Python packages?
It is recommended that you see the [full installation instructions](http://dlatk.wwbp.org/install.html#dependencies). 

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