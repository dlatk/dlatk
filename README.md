# Differential Language Analysis ToolKit

DLATK is an end to end human text analysis package for Python 3. It is specifically *suited for social media, Psychology, and health research*, developed originally for projects out of the University of Pennsylvania and Stony Brook University.  Currently, it has been used in over 75 peer-reviewed publicaitons (many from before there was an article to reference).

DLATK can perform:

- linguistic feature extraction (i.e. turning text into variables)
- differential language analysis (i.e. finding the language that is most associated with psychological or health variables)
- wordcloud visualization
- statistical- and machine learning-based supervised prediction (regression and classification)
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

### STEP 1:  Make sure you have python3-mysqldb (if using mysql):
```sh
sudo apt-get install python3-mysqldb
sudo apt install libmysqlclient-dev  #OR for MariaDB: sudo apt-get install libmariadbclient-dev
pip3 install mysqlclient
```

### STEP 2: Install from one of these options:

#### A. GitHub
```sh
git clone https://github.com/dlatk/dlatk.git
cd dlatk
python setup.py install
```
#### B. pip
```sh
pip3 install dlatk
```

#### C. conda
```sh
conda install -c wwbp dlatk
```

#### D. Docker (from 2018; may not work well for newer versions)
Detailed Docker install instructions [here](http://dlatk.wwbp.org/tutorials/tut_docker.html).

```sh
docker run --name mysql_v5  --env MYSQL_ROOT_PASSWORD=my-secret-pw --detach mysql:5.5
docker run -it --rm --name dlatk_docker --link mysql_v5:mysql dlatk/dlatk bash
```

- [DLATK at DockerHub](https://hub.docker.com/r/dlatk/dlatk/)
- [DockerFile on GitHub](https://github.com/dlatk/dlatk-docker)


### Still didn't work? If using linux, Try this:
```sh
sudo apt install python3-pip
pip3 install numpy scipy scikit-learn statsmodels jsonrpclib simplejson nltk
sudo apt-get install python3-mysqldb
sudo apt install libmysqlclient-dev
pip3 install mysqlclient
git clone https://github.com/dlatk/dlatk.git
cd dlatk
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

## Quick Start

To check if it will run:

```sh
python3 dlatkInterface.py -h
```

To add packaged data to mysql and text with it:

```sh
mysql -e 'CREATE DATABASE dla_tutorial'; cat dlatk/data/dla_tutorial.sql | mysql dla_tutorial
mysql -e 'CREATE DATABASE dlatk_lexica'; cat dlatk/data/dlatk_lexica.sql | mysql dlatk_lexica

python3 dlatkInterface.py -d dla_tutorial -t msgs -g user_id --add_ngrams -n 1 --add_lex -l dd_intAff --weighted_lex
```

Expected output:

```console
-----
DLATK Interface Initiated: XXXX-XX-XX XX:XX:XX
-----
SQL QUERY: DROP TABLE IF EXISTS feat$1gram$msgs$user_id$16to16
SQL QUERY: CREATE TABLE feat$1gram$msgs$user_id$16to16 ( id BIGINT(16) UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, group_id int(10) unsigned, feat VARCHAR(36) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin, value INTEGER, group_norm DOUBLE, KEY `correl_field` (`group_id
SQL QUERY: DROP TABLE IF EXISTS feat$meta_1gram$msgs$user_id$16to16
SQL QUERY: CREATE TABLE feat$meta_1gram$msgs$user_id$16to16 ( id BIGINT(16) UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, group_id int(10) unsigned, feat VARCHAR(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin, value INTEGER, group_norm DOUBLE, KEY `correl_field` (`gro
finding messages for 1000 'user_id's
SQL QUERY: ALTER TABLE feat$1gram$msgs$user_id$16to16 DISABLE KEYS
Messages Read: 5k
...
Messages Read: 30k
Done Reading / Inserting.
Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).
SQL QUERY: ALTER TABLE feat$1gram$msgs$user_id$16to16 ENABLE KEYS
Done

Intercept detected 5.037105 [category: AFFECT_AVG]
Intercept detected 2.399763 [category: INTENSITY_AVG]
SQL QUERY: DROP TABLE IF EXISTS feat$cat_dd_intAff_w$msgs$user_id$1gra
SQL QUERY: CREATE TABLE feat$cat_dd_intAff_w$msgs$user_id$1gra ( id BIGINT(16) UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, group_id int(10) unsigned, feat VARCHAR(13) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin, value INTEGER, group_norm DOUBLE, KEY `correl_field` (`
WORD TABLE feat$1gram$msgs$user_id$16to16
SQL QUERY: ALTER TABLE feat$cat_dd_intAff_w$msgs$user_id$1gra DISABLE KEYS
10 out of 1000 group Id's processed; 0.01 complete
20 out of 1000 group Id's processed; 0.02 complete
...
1000 out of 1000 group Id's processed; 1.00 complete
SQL QUERY: ALTER TABLE feat$cat_dd_intAff_w$msgs$user_id$1gra ENABLE KEYS
--
Interface Runtime: 167.67 seconds
DLATK exits with success! A good day indeed  ¯\_(ツ)_/¯.
```

## Documentation

The documentation for the latest release is at [dlatk.wwbp.org](dlatk.wwbp.org).

## Citation

If you use DLATK in your work please cite the following [paper](http://aclweb.org/anthology/D17-2010):

H. Andrew Schwartz, Salvatore Giorgi, Maarten Sap, Patrick Crutchley, Lyle Ungar, and Johannes Eichstaedt. 2017. DLATK: Differential Language Analysis ToolKit. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 55–60, Copenhagen, Denmark. Association for Computational Linguistics.

### bibtex
```
@InProceedings{DLATKemnlp2017,
  author =  "Schwartz, H. Andrew and Giorgi, Salvatore and Sap, Maarten and Crutchley, Patrick and Eichstaedt, Johannes and Ungar, Lyle",
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
