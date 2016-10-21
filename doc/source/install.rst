************
Installation
************

Python 2 vs 3
=============
DLATK is available for python 2.7 and 3.5, with the 3.5 version being the official release. The 2.7 version is fully functional (as of v1.0) but will not be maintained. Please make sure you ``pip`` is bound to Python 3 for the commands below. 

To install the Python 2.7 version run:

.. code-block:: bash

		pip install dlatk==0.6.1

Setup (Linux)
=============
**WARNING**: This will install MySQL on your computer. 

Install the required Ubuntu libraries. The requirements.sys can be found on the `DLATK GitHub page <http://www.github.com/dlatk/dlatk>`_.   The ``r-base`` package might be difficult to install and can be removed from ``requirements.sys`` if needed though this will limit some functionality. 
	
.. code-block:: bash

		xargs apt-get install < install/requirements.sys

Install via pip. Make sure pip is bound to 

.. code-block:: bash

		pip install dlatk

DLATK has been tested on Ubuntu 14.04. 

Setup (OSX with brew)
=====================
**WARNING**: This will install MySQL on your computer.

Install dependencies with brew. 

.. code-block:: bash

		brew install python mysql

Install via pip:

.. code-block:: bash

		pip install dlatk

DLATK has been tested on OSX 10.11. 

Setup (Anaconda)
================

Run the following in a Python 3.5 conda env:

.. code-block:: bash

		conda install -c wwbp dlatk

Install Sample Datasets
=======================
DLATK comes packaged with two sample databases: dla_tutorial and permaLexicon. See :doc:`datasets` for more information on the databases. To install them use the following:

.. code-block:: bash

		mysql -u username -p  < /path/to/dlatk/data/dla_tutorial.sql
		mysql -u username -p  < /path/to/dlatk/data/permaLexicon.sql

**WARNING**: if these databases already exist the above commands will add tables to the db. 

Install Optional Dependencies
=============================

Python
------

You can install the optional python dependencies with

.. code-block:: bash

		pip install image langid rpy2 wordcloud

Load NLTK corpus
----------------
Load NLTK data from the command line:

.. code-block:: bash

		python -c "import nltk; nltk.download('wordnet')"

Install Stanford Parser
-----------------------

#. Download the zip file from http://nlp.stanford.edu/software/lex-parser.shtml. 
#. Extract into ``../dlatk/Tools/StanfordParser/``. 
#. Move ``../dlatk/Tools/StanfordParser/oneline.sh`` into the folder you extracted: ``../dlatk/Tools/StanfordParser/stanford-parser-full*/``.
	
Install Tweet NLP v0.3 (ark-tweet-nlp-0.3)
------------------------------------------

#. Download the tgz file (for version 0.3) from http://www.cs.cmu.edu/~ark/TweetNLP/.
#. Extract this file into ``../dlatk/Tools/TwitterTagger/``.

(Optional) Install the IBM Wordcloud jar file. 
----------------------------------------------

The IBM wordcloud module is our default. To install this you must sign up for a IBM DeveloperWorks account and download ibm-word-cloud.jar. Place this file into ``../dlatk/lib/``. 

If you are unable to install this jar then you can use the python wordcloud module:

1. pip install wordcloud

2. Change ``wordcloud_algorithm='ibm'`` in ../dlatk/lib/wordcloud.py to ``wordcloud_algorithm='amueller'``.

Command Line Interface
======================

DLATK is run using dlatkInterface.py which is added to /usr/bin/local during the installation process. 

MySQL Configuration
===================

1. DLATK is *highly* dependent on MySQL. You must have this installed. 

2. Any calls to dlatkInterface.py will open MySQL. We assume any table with text data has the following columns:

* message: text data
* message_id: unique numeric identifier for each message

3. All lexicon tables are assumed to be in a database called permaLexicon (a sample database with this name is distributed with the release). To change this you must edit fwConstants.py: ``DEF_LEXICON_DB = 'permaLexicon'``

Dependencies
============

Python
------
* matplotlib (>=1.3.1)
* mysqlclient
* nltk (>=3.1)
* numpy
* pandas (>=0.17.1)
* python-dateutil (>=2.5.0)
* scikit-learn (>=0.17.1)
* scipy
* SQLAlchemy (>=0.9.9)
* statsmodels (>=0.6.1)

Python (optional)
-----------------
* image 
* langid (>=1.1.4)
* rpy2 (2.6.0)
* wordcloud (>=1.1.3)

Other
-----
* IBM Wordcloud (optional)
* Mallet (optional)
* Stanford Parser
* Tweet NLP

Install Issues
==============

See :doc:`install_faq` for more info. 