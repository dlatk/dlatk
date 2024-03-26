.. _tut_convokit:
===================
DLATK with ConvoKit
===================

In this tutorial we will import a ConvoKit data set, extract features, and run Differential Language Analysis (DLA).

Step 1 - Import Data
====================

Data must be formatted according to `ConvoKit standards <https://convokit.cornell.edu/documentation/data_format.html>`_. In particular, DLATK will import three tables: **utterances**, **speakers**, and **conversations**. The json files *corpus.json* and *index.json* are ignored.

ConvoKit comes packaged with `several data sets <https://convokit.cornell.edu/documentation/datasets.html>`_. We will use the `Cornell Movie–Dialogs Corpus <https://convokit.cornell.edu/documentation/movie.html>`_ for this tutorial, which you can download `here <https://zissou.infosci.cornell.edu/convokit/datasets/movie-corpus/>`_. To import the data you need to pass the path to the downloaded (and unzipped) corpus directory. 

.. code-block:: bash

	python3 dlatk/tools/importmethods.py --add_convokit /path/to/downloaded/data/movie-corpus

.. note::
    
    Note that wherever you run the above command from, DLATK will create a sqlite database file called *movie-corpus.db*. You will need the full path to this file for subsequent DLATK commands. 

You will then see the following printed to your terminal

.. code-block:: bash

    CREATE TABLE utterances (message_id VARCHAR(255), conversation_id VARCHAR(255), message VARCHAR(255), speaker VARCHAR(255), movie_id VARCHAR(255), parsed VARCHAR(255), reply_to VARCHAR(255), timestamp VARCHAR(255), vectors VARCHAR(255));
    Importing data, reading /path/to/downloaded/data/movie-corpus/utterances.jsonl file
	Wrote 10000 lines
	Wrote 20000 lines
    ...
	Wrote 300000 lines
    CREATE TABLE speakers (speaker VARCHAR(255), character_name VARCHAR(255), movie_idx VARCHAR(255), movie_name VARCHAR(255), gender VARCHAR(255), credit_pos VARCHAR(255), vectors VARCHAR(255));
    Importing data, reading /path/to/downloaded/data/movie-corpus/speakers.json file
    CREATE TABLE conversations (conversation_id VARCHAR(255), movie_idx VARCHAR(255), movie_name VARCHAR(255), release_year VARCHAR(255), rating VARCHAR(255), votes VARCHAR(255), genre VARCHAR(255), vectors VARCHAR(255));
    Importing data, reading /path/to/downloaded/data/movie-corpus/conversations.json file

You can then view your data by using sqlite from the command line:

.. code-block:: bash

    sqlite3 movie-corpus.db

    sqlite> .tables
    conversations  speakers       utterances

    sqlite> .schema utterances
    CREATE TABLE utterances (message_id VARCHAR(255), conversation_id VARCHAR(255), message VARCHAR(255), speaker VARCHAR(255), movie_id VARCHAR(255), parsed VARCHAR(255), reply_to VARCHAR(255), timestamp VARCHAR(255), vectors VARCHAR(255));

    sqlite> SELECT * FROM speakers limit 3;
    u0|BIANCA|m0|10 things i hate about you|f|4|[]
    u2|CAMERON|m0|10 things i hate about you|m|3|[]
    u3|CHASTITY|m0|10 things i hate about you|?|?|[]

Step 2 - Extract Features with DLATK
====================================

Now that your data is uploaded, you can extract features. The **utterances** table is contains the full turn-level conversation data. Two items have been renamed to follow DLATK conventions: `id` is now `message_id` and `text` is now `message`. 

DLATK allows one to extract features at various levels by simply changing the group flag :doc:`../fwinterface/fwflag_c` (i.e., the group or correl field specifies the level of feature aggregation). We have outcomes for both the speaker (in the **speakers** table) and conversations (in the **conversations** table). For this tutorial we will proceed with a *speaker* level analysis. 

First, we extract unigrams at the speaker level:

.. code-block:: bash

    dlatkInterface.py --engine sqlite -d /path/to/db/movie-corpus -t utterances -g speaker --add_ngrams -n 1

    -----
    DLATK Interface Initiated: 2024-03-26 18:32:10
    -----
    Connecting to SQLite database: /path/to/db/movie-corpus.db
    query: PRAGMA table_info(utterances)
    SQL Query: DROP TABLE IF EXISTS feat$1gram$utterances$speaker
    SQL Query: CREATE TABLE feat$1gram$utterances$speaker ( id INTEGER PRIMARY KEY, group_id VARCHAR(255), feat VARCHAR(36), value INTEGER, group_norm DOUBLE)
    ...

    finding messages for 8890 'speaker's
    [0%] Inserted 401 total ngram rows covering 1 speakers
    Messages Read: 5k
    Messages Read: 10k
    Messages Read: 15k
    [5%] Inserted 78283 total ngram rows covering 445 speakers
    ...
     [95%] Inserted 1356583 total ngram rows covering 8446 speakers
    Messages Read: 285k
    Messages Read: 290k
    Messages Read: 295k
    Done Reading / Inserting.
    Adding Keys (if goes to keycache, then decrease MAX_TO_DISABLE_KEYS or run myisamchk -n).
    Done

    -------
    Settings:

    Database - /path/to/db/movie-corpus
    Corpus - utterances
    Group ID - speaker
    Feature table(s) - feat$1gram$utterances$speaker
    -------
    Interface Runtime: 1390.21 seconds
    DLATK exits with success! A good day indeed  ¯\_(ツ)_/¯.


Now we have unigrams extracted for each speaker in the corpus. We can view this table using the command line: 

.. code-block:: bash

    sqlite3 movie-corpus.db

    sqlite> .tables
    conversations                       speakers                          
    feat$1gram$utterances$speaker       utterances                        
    feat$meta_1gram$utterances$speaker

    sqlite> SELECT * FROM feat$1gram$utterances$speaker LIMIT 5;
    1|u0|they|1|0.000859845227858985
    2|u0|do|3|0.00257953568357696
    3|u0|not|11|0.00945829750644884
    4|u0|!|9|0.00773860705073087
    5|u0|i|44|0.0378331900257954

To extract features at the utterance or conversation level you simply change the :doc:`../fwinterface/fwflag_c` flag:

.. code-block:: bash

    dlatkInterface.py --engine sqlite -d /path/to/db/movie-corpus -t utterances -g message_id --add_ngrams -n 1

    dlatkInterface.py --engine sqlite -d /path/to/db/movie-corpus -t utterances -g conversation_id --add_ngrams -n 1

Next, we will remove rare features. This next command will remove features which are used by less than 5% of groups (i.e., speakers). We set the :doc:`../fwinterface/fwflag_group_freq_thresh` flag to 0 so that we include all speakers.  


.. code-block:: bash

    dlatkInterface.py --engine sqlite -d /path/to/db/movie-corpus -t utterances -g speaker -f 'feat$1gram$utterances$speaker' --feat_occ_filter --set_p_occ 0.05 --group_freq_thresh 0

    -----
    DLATK Interface Initiated: 2024-03-26 19:05:26
    -----
    Connecting to SQLite database: /path/to/db/movie-corpus.db
    feat$1gram$utterances$speaker [threshold: 444]
    SQL Query: DROP TABLE IF EXISTS feat$1gram$utterances$speaker$0_05
    feat$1gram$utterances$speaker <new table feat$1gram$utterances$speaker$0_05 will have 561 distinct features.>
    SQL Query: CREATE TABLE feat$1gram$utterances$speaker$0_05 ( id INTEGER PRIMARY KEY, group_id VARCHAR(255), feat VARCHAR(36), value INTEGER, group_norm DOUBLE)
    0.1m feature instances written
    ...
    0.8m feature instances written
    Done inserting.
    Enabling keys.
    done.
    -------
    Settings:

    Database - /path/to/db/movie-corpus
    Corpus - utterances
    Group ID - speaker
    Feature table(s) - feat$1gram$utterances$speaker$0_05
    -------
    Interface Runtime: 6.44 seconds
    DLATK exits with success! A good day indeed  ¯\_(ツ)_/¯.

Step 2 - Correlate Features with Outcomes
=========================================

Here we will look at words which are used differentially across genders. Before doing this, we need to clean the gender data in the **speakers** table. 

.. code-block:: bash

    sqlite> select distinct gender from speakers;
    f
    m
    ?
    M
    F
    sqlite> update speakers set gender = 'f' where gender = 'F';
    sqlite> update speakers set gender = 'm' where gender = 'M';
    sqlite> update speakers set gender = null where gender = '?';
    sqlite> select distinct gender from speakers;
    f
    m

Now we can use DLATK to correlate unigram features with the binary gender outcome. We will perform DLA using a logistic regression and will visualize these correlations with a wordcloud. The :doc:`../fwinterface/fwflag_categorical` will convert the text string in the *gender* column to a one-hot encoding, where females are 1 and males are 0. Null entries are dropped.


.. code-block:: bash

    dlatkInterface.py --engine sqlite -d /path/to/db/movie-corpus -t utterances -g speaker \ 
    -f 'feat$1gram$utterances$speaker$0_5' \ 
    --outcome_table speakers --outcomes gender --categorical gender \ 
    --correlate --logistic_regression --tagcloud --make_wordclouds \ 
    --output gender_wordclouds

    ...
    Yielding data over ['gender__f'], adjusting for: [].
    Yielding norms with zeros (1073 groups * 561 feats).
        200 features correlated
        400 features correlated
    ...
    outputting tagcloud to: gender_wordclouds_tagcloud.txt
    Wordcloud created at: gender_wordclouds_tagcloud_wordclouds/gender__f_pos.B_0.164-0.507_wc.png
    Wordcloud created at: gender_wordclouds_tagcloud_wordclouds/gender__f_neg.B_0.167-0.470_wc.png

From the above output we see that we are correlating 561 features across 1073 observations (speakers). The names of the files show use the full range of coefficient values, e.g. 0.164 to 0.507 for the words in the positive wordcloud. Wordclouds are shown below:

.. |gender_ck_1pos| image:: ../../_static/gender_ck_1pos.png
.. |gender_ck_1neg| image:: ../../_static/gender_ck_1neg.png

============   =============================   ===========================
Outcome        Positive Correlation (female)   Negative Correlation (male)
============   =============================   ===========================
Gender         |gender_ck_1pos|                |gender_ck_1neg|
============   =============================   ===========================