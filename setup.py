import os
import sys
import subprocess
from setuptools import setup

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

The documentation for the latest release is at [https://dlatk.wwbp.org](https://dlatk.wwbp.org).

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

DISTNAME = "dlatk"
VERSION = "1.4.1"

PACKAGES = [
    "dlatk",
    "dlatk.database",
    "dlatk.lib",
    "dlatk.lexicainterface",
    "dlatk.mysqlmethods",
    "dlatk.sqlitemethods",
    "dlatk.tools",
]

LICENSE = "GNU General Public License v3 (GPLv3)"
AUTHOR = "H. Andrew Schwartz, Salvatore Giorgi, Maarten Sap, Patrick Crutchley, Lukasz Dziurzynski, Megha Agrawal, and Shashanka Subrahmanya"
EMAIL = "has@cs.stonybrook.edu, sgiorgi@sas.upenn.edu"
MAINTAINER = "Salvatore Giorgi, H. Andrew Schwartz"
MAINTAINER_EMAIL = "sgiorgi@sas.upenn.edu, has@cs.stonybrook.edu"
URL = "https://dlatk.wwbp.org"
DOWNLOAD_URL = "https://github.com/dlatk/dlatk"

CLASSIFIERS = [
    "Environment :: Console",
    "Natural Language :: English",
    "Intended Audience :: End Users/Desktop",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering",
]

PACKAGE_DATA = {
    "dlatk": ["data/*.sql", "data/*.db", "data/*.csv", "tools/colabify.sh"],
    "dlatk.lib": ["meloche_bd.ttf", "oval_big_mask.png", "oval_mask.png"],
}

INCLUDE_PACKAGE_DATA = True

# Core requirements with environment markers to support both old and new Pythons
INSTALL_REQUIRES = [
    "nltk>=3.7,<4.0",
    "python-dateutil>=2.5.0,<3.0",

    # NumPy
    'numpy>=1.20,<2.0; python_version<"3.12"',
    'numpy>=2.0; python_version>="3.12"',

    # pandas
    'pandas>=1.2,<2.0; python_version<"3.12"',
    'pandas>=2.1,<3.0; python_version>="3.12"',

    # patsy
    "patsy>=0.5.1,<=0.5.6",

    # scikit-learn
    'scikit-learn>=1.0,<=1.1.3; python_version<"3.12"',
    'scikit-learn>=1.4; python_version>="3.12"',

    # SciPy
    'scipy>=1.8,<=1.11.4; python_version<"3.12"',
    'scipy>=1.11; python_version>="3.12"',

    # statsmodels
    "statsmodels>=0.13,<0.15",
]

EXTRAS_REQUIRE = {
    "dlatk-pymallet": ['dlatk-pymallet==1.0.0; python_version<"3.12"'],
    "gensim": ["gensim"],
    "image": ["image<=1.5.33"],
    "jsonrpclib-pelix": ["jsonrpclib-pelix>=0.2.8"],
    "langid": ["langid>=1.1.4,<=1.1.6"],
    "matplotlib": [
        'matplotlib>=3.5,<3.8; python_version<"3.12"',
        'matplotlib>=3.8,<4; python_version>="3.12"',
    ],
    "rpy2": ['rpy2<3.6; python_version<"3.12"'],
    "simplejson": ["simplejson>=3.3.1"],
    "textstat": ["textstat>=0.6.1"],
    "wordcloud": ["wordcloud>=1.1.3,<=1.9.3"],
}

SCRIPTS = ["dlatkInterface.py"]

# Optionally add MySQL deps if the mysql CLI is present
if os.getenv("COLAB_RELEASE_TAG") is None:
    try:
        subprocess.check_output(["mysql", "--version"])
        INSTALL_REQUIRES += [
            'mysqlclient<=2.1.1; python_version<"3.12"',
            'mysqlclient>=2.2.0; python_version>="3.12"',
            "SQLAlchemy>=1.4,<=2.0.20",
        ]
    except Exception:
        print(
            "\\nMySQL is not installed. Skipping mysqlclient/SQLAlchemy. "
            "Install MySQL (or MariaDB) first, then:\\n"
            "  pip install 'mysqlclient' 'SQLAlchemy>=1.4,<=2.0.20'\\n"
        )

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        author=AUTHOR,
        author_email=EMAIL,
        version=VERSION,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        include_package_data=INCLUDE_PACKAGE_DATA,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        scripts=SCRIPTS,
        python_requires=">=3.8",
    )

    # Preserve the original "Colab convenience" cloning safely
    clone_folder = "/content" if os.getenv("COLAB_RELEASE_TAG") is not None else os.path.expanduser("~")
    if not os.path.exists(os.path.join(clone_folder, "dlatk")):
        try:
            subprocess.run(["git", "clone", f"{DOWNLOAD_URL}.git", os.path.join(clone_folder, "dlatk")], check=False)
        except Exception:
            pass

    if os.getenv("COLAB_RELEASE_TAG") is not None:
        try:
            import dlatk  # noqa: F401
            dlatk_path = __import__("dlatk").__path__[0]
            subprocess.run(["bash", os.path.join(dlatk_path, "tools", "colabify.sh"), dlatk_path], check=False)
        except Exception as e:
            print(f"[Colabify skipped] {e}")

