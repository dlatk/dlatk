.. _tut_lda:
====================
Mallet LDA Interface
====================

For a conceptual overview of LDA, see `this intro <http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/>`_

Step 0: Get Access To Mallet
----------------------------

* This tutorial uses `Mallet <http://mallet.cs.umass.edu/>`_. 
* Install to your home directory using the following website: http://mallet.cs.umass.edu/download.php
* NOTE - if you plan to be running on large datasets (~15M FB messages or similar) you may have to adjust parameters in your mallet script file.  See more info in the "Run LDA with Mallet" step.

Step 1: (If necessary) Create sample tweet table
------------------------------------------------
If necessary create a message table to run LDA on

.. code-block:: mysql

	use dla_tutorial; 
	create table msgs_lda like msgs;
	insert into msgs_lda select * from msgs where rand()<(2/6);

We will create an output folder in your home directory

.. code-block:: bash
	
	mkdir ~/lda_tutorial


Step 2, option A: Create tokenized table in mySQL and Export it
------------------------------------------------------------------
Using the infrastructure to tokenize the messages and print the tokenized messages in a file

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_lda -c message_id --add_tokenized   #creates table *msgs_lda_tok* in *dla_tutorial*

	dlatkInterface.py -d dla_tutorial -t msgs_lda --print_tokenized_lines ~/lda_tutorial/msgs_lda.txt

NOTE - if you have newlines in your message this step may cause issues down the line, it is recommended to replace all newlines with a representative character before proceeding

OPTIONAL: If you want to restrict your LDA words to a subset of words in the corpus (e.g., those said by at least N people, or said in no more than Y percent of messages), you can use the :doc:`../fwinterface/fwflag_feat_whitelist` flag, eg:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_lda --print_tokenized_lines ser1_filt.txt --feat_whitelist 'feat$1gram$msgs_lda$user_id$16to16$0_005'


Step 2, option B: Generate a feature table and convert it to a mallet-appropriate formatted text file
-----------------------------------------------------------------------------------------------------
You can use this step in place of step 2, option A.

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_lda -c message_id --add_ngrams -n 1

	dlatkInterface.py -d dla_tutorial -t msgs_lda -c message_id -f 'feat$1gram$msgs_lda$message_id$16to16' --print_joined_feature_lines msgs_lda.txt


Step 3: Format for Mallet
-------------------------
Prepares the messages and tokenizes them again for Mallet, removing stopwords and non English characters. For help use `./bin/mallet import-file --help`

.. code-block:: bash

	./bin/mallet import-file --input ~/lda_tutorial/msgs_lda.txt \ 
	--token-regex "(#|@)?(?!(\W)\2+)([a-zA-Z\_\-\'0-9\(-\@]{2,})" \ 
	--output ~/lda_tutorial/msgs_lda.mallet \ 
	--remove-stopwords --keep-sequence [--extra-stopwords EXTRA_STOPWORDS_FILE]

Step 4: Run LDA with Mallet
---------------------------
This is the actual LDA step, which might take a while (4 days and a half on 20 mil tweets) for help do `./bin/mallet train-topics --help`

.. code-block:: bash

	./bin/mallet train-topics --input  ~/lda_tutorial/msgs_lda.mallet \ 
	--alpha 5 --num-topics 2000 --optimize-burn-in 0 --output-model ~/lda_tutorial/msgs_lda.model \ 
	--output-state ~/lda_tutorial/msgs_lda_state.gz \ 
	--output-topic-keys ~/lda_tutorial/msgs_lda.keys

Here **alpha** is a prior on he number of topics per document. The other hyper-parameter **beta** (which we usually do not change) is a prior on the number of words per topic.

This creates the following files:

* 

*Note*: When dealing with giant sets of data, for example creating Facebook topics, one might encounter the error **Exception in thread "main" java.lang.OutOfMemoryError: Java heap space**. You must edit the following line in **~/Mallet/bin/mallet**: *MEMORY=1g*. You can then change the 1g value upwards – to 2g, 4g, or even higher depending on your system’s RAM, which you can find out by looking up the machine’s system information.

Step 5: Add message ID’s to state file
--------------------------------------
Adds the message ID’s to the topic distributions and stores the result in lda_topics

.. code-block:: bash

	dlatkInterface.py --add_message_id ~/lda_tutorial/msgs_lda.txt ~/lda_tutorial/msgs_lda_state.gz --output_name ~/lda_tutorial/lda_topics

Step 6: Import state file into database
---------------------------------------
Imports the topic-message probability distributions in a raw format (type of JSON) not readable by DLA

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_lda_tok --add_lda_messages  ~/lda_tutorial/lda_topics

This creates the table **msgs_lda_tok_lda$lda_topics** in the database dla_tutorial.

NOTE - "Duplicate entry 'xxxx' for key 'PRIMARY'" errors may be indicative of an issues with newlines.  See step 2 for a solution.

Step 7: Create topic-word distributions
---------------------------------------
Creates the readable distributions on the messages

.. code-block:: bash

	python dlatk/LexicaInterface/topicExtractor.py -d dla_tutorial -t msgs_lda -m 'msgs_lda_tok_lda$lda_topics' --create_dists

This creates the following files:

* msgs_lda_tok_lda.lda_topics.freq.threshed50.loglik.csv
* msgs_lda_tok_lda.lda_topics.lik.csv
* msgs_lda_tok_lda.lda_topics.loglik.csv
* msgs_lda_tok_lda.lda_topics.topicGivenWord.csv
* msgs_lda_tok_lda.lda_topics.wordGivenTopic.csv

Step 8: Add topic-lexicon to lexicon database
---------------------------------------------
Generates the lexicons based on different probability distribution types

* topic given word 

.. code-block:: bash

	python dlatk/LexicaInterface/lexInterface.py --topic_csv \ 
	--topicfile=~/lda_tutorial/msgs_lda_tok_lda.lda_topics.topicGivenWord.csv \ 
	-c msgs_lda_cp

* frequency, thresholded to loglik >= 50

.. code-block:: bash

	python dlatk/LexicaInterface/lexInterface.py --topic_csv \ 
	--topicfile=~/lda_tutorial/msgs_lda_tok_lda.lda_topics.freq.threshed50.loglik.csv \ 
	-c msgs_lda_freq_t50ll 


Step 9: Extract features from lexicon
--------------------------------------
You’re now ready to start using the topic distribution lexicon

.. code-block:: bash

	dlatkInterface.py -d DATABASE -t MESSAGE_TABLE --add_lex_table -l msgs_lda_cp --weighted_lexicon -c GROUP_ID

(always extract features using the _cp lexicon. The “freq_t50ll” lexicon is only used when generating topic_tagclouds: :doc:`../fwinterface/fwflag_topic_tagcloud` :doc:`../fwinterface/fwflag_topic_lexicon` ...freq_t50ll”)