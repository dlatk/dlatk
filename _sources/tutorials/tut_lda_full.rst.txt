.. _tut_lda:
====================
Mallet LDA Interface
====================

**NOTE: DLATK can now produce LDA topics through a simpler interface:** :doc:`tut_lda`\ **. Although the steps described below are still valid, you are advised to use the new automated interface.**

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

This will create the file `~/lda_tutorial/msgs_lda.txt`:

.. code-block:: bash
	> head ~/lda_tutorial/msgs_lda.txt
	11 en urllink kitty's claws are digging into tim's legs ... note the expression ... urllink
	12 en urllink an up close and personal encounter between pup and kitty . a few moments after this photo , pup's ass was kicked by what i can only assume to be puss n boot's cousin ... urllink
	22 en urllink here i am on the " all you can drink rum cruise " outside of nassau . man , those were the days ... urllink
	28 en urllink the blue eye .. natural light . urllink

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

This will create the file `~/lda_tutorial/msgs_lda.mallet`.

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

* msgs_lda.model
* msgs_lda_state.gz
* msgs_lda.keys

The file `msgs_lda.keys` contains the topics and at this point is it good to inspect them to see if you should change any of the above parameters (number of topics, alpha, beta, etc.)

.. code-block:: bash

	> head ~/lda_tutorial/msgs_lda.keys
	0       0.0025  italian pig americans wolf american straw pigs 04 stick film defamation censored pinocchio brick 03 sopranos pig's offensive june wop 
	1       0.0025  yoga levi yea pages yay morning boy mix lorraine evening bit law she's joe honey exam property study gay spring 
	2       0.0025  projection sng recap mtt dried 18 sessions planetpoker 5.50 pokerstars faulty limit summary music-lovers end-users 1540-1608 raritan tester dch brigade 
	3       0.0025  :) urllink misses marion lovely managed melbourne sean pub sort round rock bus :( tent ;) perth we've elephant flat 
	4       0.0025  jill alcohol aaaannnndd soorry georgie celica precariously timid mirrors cam defined cat's praying expects hitler valley asses 
	5       0.0025  apc uhm downsides surpression tly pary undies spybot band-aids pow sprinkles trashed bloated celebrated paramaters thas elses country's immortal association 

*Note*: When dealing with giant sets of data, for example creating Facebook topics, one might encounter the error **Exception in thread "main" java.lang.OutOfMemoryError: Java heap space**. You must edit the following line in **~/Mallet/bin/mallet**: *MEMORY=1g*. You can then change the 1g value upwards – to 2g, 4g, or even higher depending on your system’s RAM, which you can find out by looking up the machine’s system information.

Step 5: Add message ID’s to state file
--------------------------------------
Adds the message ID’s to the topic distributions and stores the result in lda_topics

.. code-block:: bash

	dlatkInterface.py --add_message_id ~/lda_tutorial/msgs_lda.txt ~/lda_tutorial/msgs_lda_state.gz --output_name ~/lda_tutorial/lda_topics

This creates the file `~/lda_tutorial/lda_topics`:

.. code-block:: bash

	> head ~/lda_tutorial/lda_topics
	#doc source pos typeindex type topic
	#alpha : 0.0025 0.0025 0.0025 0.0025 ....
	#beta : 0.01
	0 2 0 0 miss 1840
	0 2 1 1 su 661   
	0 2 2 2 living 623
	0 2 3 3 skills 466
	0 2 4 4 teacher 466
	0 2 5 5 form 1319

Step 6: Import state file into database
---------------------------------------
Imports the topic-message probability distributions in a raw format (type of JSON) not readable by DLA

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_lda_tok --add_lda_messages  ~/lda_tutorial/lda_topics

This creates the table **msgs_lda_tok_lda$lda_topics** in the database dla_tutorial:

.. code-block:: mysql

	select message from msgs_lda_tok_lda$lda_topics limit 1;
	| [{"topic_id": "296", "doc": "10", "term": "urllink", "index": "0", "term_id": "191", "message_id": "40"}, {"topic_id": "947", "doc": "10", "term": "busy", "index": "1", "term_id": "249", "message_id": "40"}, {"topic_id": "1804", "doc": "10", "term": "roadway", "index": "2", "term_id": "250", "message_id": "40"}, {"topic_id": "296", "doc": "10", "term": "urllink", "index": "3", "term_id": "191", "message_id": "40"}]

*NOTE* - "Duplicate entry 'xxxx' for key 'PRIMARY'" errors may be indicative of an issues with newlines.  See step 2 for a solution.

Step 7: Create topic-word distributions
---------------------------------------
Creates the readable distributions on the messages

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_lda -m 'msgs_lda_tok_lda$lda_topics' --create_dists

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

	dlatkInterface.py --lex_interface --topic_csv  \ 
	--topicfile=/home/user/lda_tutorial/msgs_lda_tok_lda.lda_topics.topicGivenWord.csv  \ 
	-c msgs_lda_cp

This will create the table `msgs_lda_cp` in the database `dlatk_lexica`. You can change the database with the flag :doc:`../fwinterface/fwflag_lexicondb`:

.. code-block:: bash

	mysql> select * from msgs_lda_cp limit 5;
	+----+----------------+----------+--------------------+
	| id | term           | category | weight             |
	+----+----------------+----------+--------------------+
	|  1 | erm            | 274      |             0.0625 |
	|  2 | productively   | 274      |  0.333333333333333 |
	|  3 | jared          | 274      |              0.125 |
	|  4 | book           | 274      |                  1 |
	|  5 | sketch         | 274      | 0.0909090909090909 |
	+----+----------------+----------+--------------------+


* frequency, thresholded to loglik >= 50

.. code-block:: bash

	dlatkInterface.py --lex_interface --topic_csv \ 
	--topicfile=~/lda_tutorial/msgs_lda_tok_lda.lda_topics.freq.threshed50.loglik.csv \ 
	-c msgs_lda_freq_t50ll 

This will create the table `msgs_lda_freq_t50ll` in the database `dlatk_lexica`. You can change the database with the flag :doc:`../fwinterface/fwflag_lexicondb`:

.. code-block:: bash

	mysql> select * from msgs_lda_freq_t50ll limit 5;
	+----+------------------+----------+--------+
	| id | term             | category | weight |
	+----+------------------+----------+--------+
	|  1 | wildest          | 766      |      1 |
	|  2 | kazadoom         | 766      |      1 |
	|  3 | sentirnos        | 766      |      1 |
	|  4 | charlie          | 766      |      1 |
	|  5 | commondreams.org | 766      |      1 |
	+----+------------------+----------+--------+

Step 9: Extract features from lexicon
--------------------------------------
You’re now ready to start using the topic distribution lexicon

.. code-block:: bash

	dlatkInterface.py -d DATABASE -t MESSAGE_TABLE -c GROUP_ID --add_lex_table -l msgs_lda_cp --weighted_lexicon 

(always extract features using the _cp lexicon. The “freq_t50ll” lexicon is only used when generating topic_tagclouds: :doc:`../fwinterface/fwflag_topic_tagcloud` :doc:`../fwinterface/fwflag_topic_lexicon`)
