************
Installation
************

Python 2 vs 3
=============
DLATK is available for python 2.7 and 3.5, with the 3.5 version being the official release. The 2.7 version is fully functional (as of v1.0) but will not be maintained. 


Setup (Linux)
=============
DLATK has been tested on Ubuntu 14.04. **WARNING**: Step 1 will install MySQL on your computer. 

Install the required Ubuntu libraries. The package python-rpy2 can also be omitted if you experience installation issues, thought his will limit some of the advanced methods in DLATK.
 	
.. code-block:: bash

 		xargs apt-get install < install/requirements.sys

Install python dependencies.

.. code-block:: bash

    	pip install dlatk

Setup (OSX with brew)
=====================
DLATK has been tested on OSX 10.11.

Install dependencies with brew. WARNING: This will install MySQL on your computer.

.. code-block:: bash

    	brew install python mysql

Install python dependencies. Note that there is an issue when running this on OSX El Capitan which can be fixed by adding '--ignore-installed six' to the end of the following command. This issue does not seem to be an issue if anaconda is installed.

.. code-block:: bash

    	pip install dlatk

Setup (Anaconda)
================


Install NLTK, Stanford Parser, Tweet NLP and wordcloud
======================================================

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

The default wordcloud module is installed in Step 2 via pip. This can be changed to the IBM wordcloud module which produces nicer wordclouds. To do this:

1. You must sign up for a IBM DeveloperWorks account and download ibm-word-cloud.jar. Place this file into ``../dlatk/lib/``. 

2. Change the  ../dlatk/lib/wordcloud.py to ``wordcloud_algorithm='ibm'``

MySQL Configuration
===================

1. DLATK is *highly* dependent on MySQL. You must have this installed (see Step 1 in SETUP (Linux) or Step 3 in SETUP (OSX)). 

2. Any calls to fwInterface.py will open MySQL. With your database any text data must have two columns:

* message: text data
* message_id: unique numeric identifier for each message

3. All lexicon tables are assumed to be in a database called permaLexicon. To change this you must edit fwInterface.py: ``DEF_LEXICON_DB = 'permaLexicon'``

Dependencies
============

Python
------
* image
* matplotlib
* mysqlclient
* nltk
* numpy
* pandas
* python-dateutil
* rpy2
* scikit-learn
* scipy
* SQLAlchemy
* statsmodels
* wordcloud

Other
--------
* IBM Wordcloud
* Mallet
* Stanford Parser
* Tweet NLP
