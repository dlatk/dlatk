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

DLATK is available via [docker](https://github.com/pennsignals/dlatk-docker) or through conda, pip or github via:

```sh
conda install -c wwbp dlatk
```

```sh
pip install dlatk
```

```sh
python setup.py install
```

## Dependencies
- [mysqlclient](https://github.com/PyMySQL/mysqlclient-python)
- [NumPy](http://www.numpy.org)
- [scikit-learn](http://www.scikit-learn.org/)
- [SciPy](http://www.scipy.org/)
- [statsmodels](http://www.statsmodels.org/)

See the [full installation instructions](http://dlatk.wwbp.org/install.html#dependencies)
for recommended and optional dependencies.

## Documentation

The documentation for the latest release is at [dlatk.wwbp.org](http://dlatk.wwbp.org).

## Citation

If you use DLATK in your work please cite the following paper:

```
@InProceedings{DLATKemnlp2017,
  author = 	"Schwartz, H. Andrew
		and Giorgi, Salvatore
		and Sap, Maarten
		and Crutchley, Patrick
		and Eichstaedt, Johannes
		and Ungar, Lyle",
  title = 	"DLATK: Differential Language Analysis ToolKit",
  booktitle = 	"Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
  year = 	"2017",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"55--60",
  location = 	"Copenhagen, Denmark",
  url = 	"http://aclweb.org/anthology/D17-2010"
}

```

## License

Licensed under a [GNU General Public License v3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Background

Developed by the [World Well-Being Project](http://www.wwbp.org) based out of the University of Pennsylvania and Stony Brook University.
