import sys
import os
import subprocess

_version = sys.version_info[0]

try:
  from setuptools import setup
except ImportError:
  if _version >= 3:
    sys.exit("Need setuptools to install dlatk for Python 3.x")
  from distutils.core import setup


DESCRIPTION = """DLATK is an end to end human text analysis package, specifically suited for social media and social scientific applications. It is written in Python 3 and developed by the World Well-Being Project at the University of Pennsylvania and Stony Brook University. """
LONG_DESCRIPTION = """
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
"""
DISTNAME = 'dlatk'
PACKAGES = ['dlatk',
  'dlatk.lib',
  'dlatk.lexicainterface',
  'dlatk.mysqlmethods',
  'dlatk.tools',
]
LICENSE = 'GNU General Public License v3 (GPLv3)'
AUTHOR = "H. Andrew Schwartz, Salvatore Giorgi, Maarten Sap, Patrick Crutchley, Lukasz Dziurzynski and Megha Agrawal"
EMAIL = "has@cs.stonybrook.edu, sgiorgi@sas.upenn.edu"
MAINTAINER = "Salvatore Giorgi, H. Andrew Schwartz"
MAINTAINER_EMAIL = "sgiorgi@sas.upenn.edu, has@cs.stonybrook.edu"
URL = "http://dlatk.wwbp.org"
DOWNLOAD_URL = 'https://github.com/dlatk/dlatk'
CLASSIFIERS = [
  'Environment :: Console',
  'Natural Language :: English',
  'Intended Audience :: End Users/Desktop',
  'Intended Audience :: Developers',
  'Intended Audience :: Science/Research',
  'Programming Language :: Python',
  'Programming Language :: Python :: 3',
  'Programming Language :: Python :: 3.5',
  'Topic :: Scientific/Engineering',
]
VERSION = '1.1.8'
PACKAGE_DATA = {
  'dlatk': ['data/*.sql'],
  'dlatk.lib': ['lib/meloche_bd.ttf', 'lib/oval_big_mask.png', 'lib/oval_mask.png'],
}
INCLUDE_PACKAGE_DATA = True
SETUP_REQUIRES = [
  'numpy', 
]
INSTALL_REQUIRES = [
  'matplotlib>=1.3.1', 
  'mysqlclient', 
  'nltk>=3.2.1', 
  'numpy', 
  'pandas>=0.20.3', 
  'patsy>=0.2.1', 
  'python-dateutil>=2.5.0', 
  'scikit-learn==0.18.2', 
  'scipy>=0.19.1', 
  'SQLAlchemy>=1.0.13', 
  'statsmodels>=0.8.0', 
]
EXTRAS_REQUIRE = {
  'image': ['image'],
  'jsonrpclib-pelix': ['jsonrpclib-pelix>=0.2.8'],
  'langid': ['langid>=1.1.4'],
  'rpy2': ['rpy2'],
  'simplejson': ['simplejson>=3.3.1'],
  'textstat': ['textstat>=0.6.1'],
  'wordcloud':  ['wordcloud==1.1.3'],
}

SCRIPTS = ['dlatkInterface.py']

if __name__ == "__main__":

  setup(name=DISTNAME,
      author=AUTHOR,
      author_email=EMAIL, 
      version=VERSION,
      packages=PACKAGES,
      package_data=PACKAGE_DATA,
      include_package_data=INCLUDE_PACKAGE_DATA,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      license=LICENSE,
      url=URL,
      download_url=DOWNLOAD_URL,
      classifiers=CLASSIFIERS,
      setup_requires=SETUP_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      install_requires=INSTALL_REQUIRES,
      scripts = SCRIPTS,
      long_description_content_type ='text/markdown',
  )

