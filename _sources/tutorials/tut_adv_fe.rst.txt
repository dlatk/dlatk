.. _tut_adv_fe:
===========================
Advanced Feature Extraction
===========================

This is an overview of the different feature extraction methods. Each method needs the following flags:

* :doc:`../fwinterface/fwflag_d`: the database we are using
* :doc:`../fwinterface/fwflag_t`: the table inside the database where our text lives (aka the message table)
* :doc:`../fwinterface/fwflag_c`: the table column we will be grouping the text by (aka group)

We start with a message table called "msgs" (available in the packaged data):

.. code-block:: mysql 

   mysql> describe msgs;
   +--------------+------------------+------+-----+---------+----------------+
   | Field        | Type             | Null | Key | Default | Extra          |
   +--------------+------------------+------+-----+---------+----------------+
   | message_id   | int(11)          | NO   | PRI | NULL    | auto_increment |
   | user_id      | int(10) unsigned | YES  | MUL | NULL    |                |
   | date         | varchar(64)      | YES  |     | NULL    |                |
   | created_time | datetime         | YES  | MUL | NULL    |                |
   | message      | text             | YES  |     | NULL    |                |
   +--------------+------------------+------+-----+---------+----------------+

N-grams, Collocations, Tf-idf and more
======================================

Unigrams
--------

* :doc:`../fwinterface/fwflag_add_ngrams`
* -n

.. code-block:: bash

   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id --add_ngrams -n 1

.. code-block:: mysql

   mysql> select * from feat$1gram$msgs$user_id limit 5;
   +----+----------+-----------+-------+----------------------+
   | id | group_id | feat      | value | group_norm           |
   +----+----------+-----------+-------+----------------------+
   |  1 |    28451 | wonderful |     1 | 0.000953288846520496 |
   |  2 |    28451 | let       |     1 | 0.000953288846520496 |
   |  3 |    28451 | promotion |     1 | 0.000953288846520496 |
   |  4 |    28451 | assured   |     1 | 0.000953288846520496 |
   |  5 |    28451 | lime      |     1 | 0.000953288846520496 |
   +----+----------+-----------+-------+----------------------+

This also creates a "meta" table called feat$meta_1gram$msgs$user_id which contains average 1gram length, average 1grams per message and total 1grams:

.. code-block:: mysql

   mysql> select * from feat$meta_1gram$msgs$user_id limit 5;
   +----+----------+------------------+-------+------------------+
   | id | group_id | feat             | value | group_norm       |
   +----+----------+------------------+-------+------------------+
   |  1 |    28451 | _avg1gramLength  |     4 | 3.76549094375596 |
   |  2 |    28451 | _avg1gramsPerMsg |    81 | 80.6923076923077 |
   |  3 |    28451 | _total1grams     |  1049 |             1049 |
   |  4 |   174357 | _avg1gramLength  |     4 | 3.73343605546995 |
   |  5 |   174357 | _avg1gramsPerMsg |   216 | 216.333333333333 |
   +----+----------+------------------+-------+------------------+

N-grams
-------

This command will make separate feature tables for each "n". 

* :doc:`../fwinterface/fwflag_add_ngrams`
* -n

.. code-block:: bash

   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id --add_ngrams -n 2 3

.. code-block:: mysql

   mysql> select * from feat$2gram$msgs$user_id limit 5;
   +----+----------+------------------+-------+----------------------+
   | id | group_id | feat             | value | group_norm           |
   +----+----------+------------------+-------+----------------------+
   |  1 |    28451 | this time        |     2 |  0.00193050193050193 |
   |  2 |    28451 | email ,          |     1 | 0.000965250965250965 |
   |  3 |    28451 | comfortable than |     1 | 0.000965250965250965 |
   |  4 |    28451 | do something     |     1 | 0.000965250965250965 |
   |  5 |    28451 | charecter ,      |     1 | 0.000965250965250965 |
   +----+----------+------------------+-------+----------------------+

   mysql> select * from feat$3gram$msgs$user_id limit 5;
   +----+----------+-------------------+-------+----------------------+
   | id | group_id | feat              | value | group_norm           |
   +----+----------+-------------------+-------+----------------------+
   |  1 |    28451 | i did something   |     1 | 0.000977517106549365 |
   |  2 |    28451 | to my old         |     1 | 0.000977517106549365 |
   |  3 |    28451 | , lots of         |     1 | 0.000977517106549365 |
   |  4 |    28451 | out some babies   |     1 | 0.000977517106549365 |
   |  5 |    28451 | stumbled across a |     1 | 0.000977517106549365 |
   +----+----------+-------------------+-------+----------------------+

N-grams From Other Tokenizers
-----------------------------

DLATK uses `Happier Fun Tokenizer <https://github.com/dlatk/happierfuntokenizing>`_ as its standard tokenizer. It also has the option of using the `TweetNLP <http://www.cs.cmu.edu/~ark/TweetNLP/>`_ tokenizer with the :doc:`../fwinterface/fwflag_add_tweettok` flag. One can go straight to a feature table from a message table, via Happier Fun Tokenizer, with :doc:`../fwinterface/fwflag_add_ngrams`. Alternatively, one can go from a tokenized table via :doc:`../fwinterface/fwflag_add_tweettok` or :doc:`../fwinterface/fwflag_add_tokenized` (or any other tokenizer you wish to use) to a feature table with :doc:`../fwinterface/fwflag_add_ngrams_from_tokenized`

.. code-block:: bash

   ./dlatkInterface.py -d dla_tutorial -t msgs_tok -g user_id --add_ngrams_from_tokenized -n 1

.. code-block:: mysql

   mysql> select * from feat$1gram$msgs_tok$user_id limit 5;
   +----+----------+-------------+-------+----------------------+
   | id | group_id | feat        | value | group_norm           |
   +----+----------+-------------+-------+----------------------+
   |  1 |    28451 | nod         |     1 | 0.000953288846520496 |
   |  2 |    28451 | pub         |    11 |   0.0104861773117255 |
   |  3 |    28451 | destruction |     1 | 0.000953288846520496 |
   |  4 |    28451 | else        |     1 | 0.000953288846520496 |
   |  5 |    28451 | ?           |     4 |  0.00381315538608198 |
   +----+----------+-------------+-------+----------------------+

Feature Occurrence Filter
-------------------------

This removes rare features. Specifically, it filters features so as to keep only those features which are used by X percentage of groups or more. The missing features are aggregated into a feature called <OOV> which contains the value and group norm data for all the missing features. The percentage X is set with the --set_p_occ flag. 


* :doc:`../fwinterface/fwflag_feat_occ_filter` 
* :doc:`../fwinterface/fwflag_set_p_occ`

.. code-block:: bash

   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id -f 'feat$1to3gram$msgs$user_id' --feat_occ_filter --set_p_occ .05 --group_freq_thresh 500

Note the use of --group_freq_thresh. This is one of the only feature extraction methods where this flag is considered.

Character n-grams
-----------------

* --add_char_ngrams
* -n

.. code-block:: bash

   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id --add_char_ngrams -n 1 

.. code-block:: bash

   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id --add_char_ngrams -n 1 2 --combine_feat_tables 1to2Cgram

.. code-block:: mysql

   mysql> select * from feat$1to2Cgram$msgs$user_id limit 5;
   +----+----------+------+-------+---------------------+
   | id | group_id | feat | value | group_norm          |
   +----+----------+------+-------+---------------------+
   |  1 |    28451 |      |   898 |   0.184659675097676 |
   |  2 |    28451 | v    |    45 | 0.00925354719309068 |
   |  3 |    28451 | d    |   125 |  0.0257042977585852 |
   |  4 |    28451 | ;    |     9 | 0.00185070943861814 |
   |  5 |    28451 | y    |    71 |  0.0146000411268764 |
   +----+----------+------+-------+---------------------+

TF-IDF Tables
-------------

Creates new feature table where the group_norm is the tf-idf score

* :doc:`../fwinterface/fwflag_tf_idf`

.. code-block:: bash

   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id -f 'feat$1gram$msgs$user_id' --tf_idf

.. code-block:: mysql

   mysql> select * from feat$tf_idf_1gram$msgs$user_id order limit 5;;
   +---------+----------+-----------+-------+--------------------+
   | id      | group_id | feat      | value | group_norm         |
   +---------+----------+-----------+-------+--------------------+
   |  307349 |  2033616 | delivered |     1 | 0.0000878334772103 |
   |  278647 |  4144593 | crap      |     6 |  0.000998442620366 |
   | 1043863 |  3482840 | story     |     2 |  0.000334689956064 |
   | 1150911 |  2876677 | uh        |     2 |  0.000141436336165 |
   |  283547 |  3711805 | crosses   |     2 |  0.000827587016091 |
   +---------+----------+-----------+-------+--------------------+

Collocations and Pointwise Mutual Information
---------------------------------------------

* :doc:`../fwinterface/fwflag_feat_colloc_filter` 
* :doc:`../fwinterface/fwflag_set_pmi_threshold`

.. code-block:: bash

   # creates the table feat$1to3gram$msgs$user_id
   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id -f 'feat$1to3gram$msgs$user_id' --feat_colloc_filter --set_pmi_threshold 6.0

Transformed Tables
------------------

These switches transform the feature table during feature extraction and therefore need least one feature extraction command: --add_ngrams, --add_lex_table, etc.

* :doc:`../fwinterface/fwflag_anscombe`
* :doc:`../fwinterface/fwflag_boolean`
* :doc:`../fwinterface/fwflag_log`
* :doc:`../fwinterface/fwflag_sqrt`

.. code-block:: bash

   # produces the table feat$1gram$msgs$user_id$16to8
   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id --add_ngrams -n 1 --anscombe

   # produces the table feat$1gram$msgs$user_id$16to4
   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id --add_ngrams -n 1 --sqrt

   # produces the table feat$1gram$msgs$user_id$16to3
   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id --add_ngrams -n 1 --log

   # produces the table feat$1gram$msgs$user_id$16to1
   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id --add_ngrams -n 1 --boolean

.. code-block:: mysql

   mysql> select * from feat$1gram$msgs$user_id$16to8 limit 5;
   +-----+----------+------+-------+---------------+
   | id  | group_id | feat | value | group_norm    |
   +-----+----------+------+-------+---------------+
   | 188 |    28451 | !    |     8 | 1.23713590324 |
   | 296 |    28451 | $    |     1 | 1.22630059748 |
   | 204 |    28451 | '    |     2 | 1.22785435243 |
   | 223 |    28451 | *    |     4 | 1.23095597872 |
   |  38 |    28451 | ,    |    53 | 1.30464448623 |
   +-----+----------+------+-------+---------------+

   mysql> select * from feat$1gram$msgs$user_id$16to4 limit 5;
   +-----+----------+------+-------+-----------------+
   | id  | group_id | feat | value | group_norm      |
   +-----+----------+------+-------+-----------------+
   | 275 |    28451 | !    |     8 | 0.0873287511199 |
   | 245 |    28451 | $    |     1 | 0.0308753760547 |
   | 414 |    28451 | '    |     2 |   0.04366437556 |
   | 239 |    28451 | *    |     4 | 0.0617507521094 |
   |  45 |    28451 | ,    |    53 |  0.224776130551 |
   +-----+----------+------+-------+-----------------+

   mysql> select * from feat$1gram$msgs$user_id$16to3 limit 5;
   +-----+----------+------+-------+-------------------+
   | id  | group_id | feat | value | group_norm        |
   +-----+----------+------+-------+-------------------+
   | 278 |    28451 | !    |     8 |  0.00759737747394 |
   | 244 |    28451 | $    |     1 | 0.000952834755272 |
   | 265 |    28451 | '    |     2 |  0.00190476248065 |
   | 171 |    28451 | *    |     4 |  0.00380590373768 |
   | 283 |    28451 | ,    |    53 |   0.0492893813166 |
   +-----+----------+------+-------+-------------------+


   mysql> select * from feat$1gram$msgs$user_id$16to1 limit 5;
   +-----+----------+------+-------+------------+
   | id  | group_id | feat | value | group_norm |
   +-----+----------+------+-------+------------+
   |  51 |    28451 | !    |     8 |          1 |
   | 148 |    28451 | $    |     1 |          1 |
   | 105 |    28451 | '    |     2 |          1 |
   | 277 |    28451 | *    |     4 |          1 |
   | 304 |    28451 | ,    |    53 |          1 |
   +-----+----------+------+-------+------------+


Word Tables
-----------

* :doc:`../fwinterface/fwflag_word_table`
* :doc:`../fwinterface/fwflag_group_freq_thresh`

The word table is used to select groups that meet a certain language useage threshold. This is what we call the "group frequency threshold", as specified by the --group_freq_thresh flag. It says that we will only consider groups who use at least N words (typically 1 when working at the message level, 500 when working at the user level and 40,000 when working with communities). The word table is automatically queried based on the -t and -g flag. For example, given the following base command:

.. code-block:: bash

   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id 

DLATK will query the table "feat$1gram$msgs$user_id". The flag --word_table overrides this. It is especially useful when working with large data when the standard word table will not fit into memory. In this case we often use a feature occurrence filtered table (filtered at a small threshold). For example

.. code-block:: bash

   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id --word_table 'feat$1gram$msgs$user_id$0_01'

Lexica
------

DLATK supports both unweighted and weighted lexica. Here is an example of an unweighted lexicon. Note that the MySQL table still contains the column "weight" which is set to 1 everywhere. This is unnecessary but sometimes more insightful to be explicit.

.. code-block:: bash

   # creates the table feat$cat_LIWC2015$msgs$user_id$1gra
   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id --add_lex_table -l LIWC2015

.. code-block:: mysql

   mysql> select * from dlatk_lexica.LIWC2015 limit 5;
   +----+------+----------+--------+
   | id | term | category | weight |
   +----+------+----------+--------+
   |  1 | he   | PPRON    |      1 |
   |  2 | he'd | PPRON    |      1 |
   |  3 | he's | PPRON    |      1 |
   |  4 | her  | PPRON    |      1 |
   |  5 | hers | PPRON    |      1 |
   +----+------+----------+--------+

   mysql> select * from feat$cat_met_a30_2000_cp_w$msgs$user_id$1gra  limit 5;
   +----+----------+------+-------+--------------------------+
   | id | group_id | feat | value | group_norm               |
   +----+----------+------+-------+--------------------------+
   |  1 |    28451 | 298  |     4 | 0.0000000217525421774642 |
   |  2 |    28451 | 278  |     6 |     0.000150407662892745 |
   |  3 |    28451 | 295  |    17 |     0.000545379245206831 |
   |  4 |    28451 | 1375 |    47 |       0.0010413347897739 |
   |  5 |    28451 | 276  |    15 |     0.000299298548129527 |
   +----+----------+------+-------+--------------------------+

Here is an example of a weighted lexicon. Note the use of the --weighted_lexicon flag. Here we are using LDA Facebook topics which are available `here <http://wwbp.org/data.html>`_).


* :doc:`../fwinterface/fwflag_add_lex_table`
* :doc:`../fwinterface/fwflag_weighted_lexicon`

.. code-block:: bash

   # creates the table feat$cat_met_a30_2000_cp_w$msgs$user_id$1gra
   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id --add_lex_table -l met_a30_2000_cp --weighted_lexicon

.. code-block:: mysql

   mysql> select * from dlatk_lexica.met_a30_2000_cp limit 5;
   +----+---------+----------+--------------------+
   | id | term    | category | weight             |
   +----+---------+----------+--------------------+
   |  1 | ce      | 344      |  0.000162284972412 |
   |  2 | concept | 344      |  0.000556947925369 |
   |  3 | cough   | 344      | 0.0000711541198235 |
   |  4 | bring   | 344      |   0.00570741964554 |
   |  5 | finest  | 344      |  0.000520020800832 |
   +----+---------+----------+--------------------+

   mysql> select * from feat$cat_met_a30_2000_cp_w$msgs$user_id$1gra  limit 5;
   +----+----------+------+-------+--------------------------+
   | id | group_id | feat | value | group_norm               |
   +----+----------+------+-------+--------------------------+
   |  1 |    28451 | 298  |     4 | 0.0000000217525421774642 |
   |  2 |    28451 | 278  |     6 |     0.000150407662892745 |
   |  3 |    28451 | 295  |    17 |     0.000545379245206831 |
   |  4 |    28451 | 1375 |    47 |       0.0010413347897739 |
   |  5 |    28451 | 276  |    15 |     0.000299298548129527 |
   +----+----------+------+-------+--------------------------+

Combining Feature Tables
------------------------

Combine multiple feature tables into a single table.

* :doc:`../fwinterface/fwflag_combine_feat_tables`

.. code-block:: bash

   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id -f 'feat$1gram$msgs$user_id' 'feat$2gram$msgs$user_id' 'feat$3gram$msgs$user_id' --combine_feat_tables 1to3gram

This also works during ngram extraction:

.. code-block:: bash

   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id --add_ngrams -n 1 2 3 --combine_feat_tables 1to3gram

Part of Speech
==============

Part of Speech Usage
--------------------

* :doc:`../fwinterface/fwflag_add_pos_table`

.. code-block:: bash

   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id --add_pos_table

.. code-block:: mysql

   mysql> select * from feat$pos$msgs$user_id limit 5;
   +----+----------+------+-------+----------------------+
   | id | group_id | feat | value | group_norm           |
   +----+----------+------+-------+----------------------+
   |  1 |  2300555 | RP   |    12 |  0.00575539568345324 |
   |  2 |  2300555 | ''   |     1 | 0.000479616306954436 |
   |  3 |  2300555 | PRP  |   107 |   0.0513189448441247 |
   |  4 |  2300555 | CC   |    61 |   0.0292565947242206 |
   |  5 |  2300555 | WRB  |    15 |  0.00719424460431655 |
   +----+----------+------+-------+----------------------+

Part of Speech N-grams
----------------------

* :doc:`../fwinterface/fwflag_add_pos_ngram_table`


.. code-block:: bash

   ./dlatkInterface.py -d dla_tutorial -t msgs -g user_id --add_pos_ngram_table

.. code-block:: mysql

   mysql> select * from feat$1gram_pos$msgs$user_id limit 5;
   +----+----------+----------------------+-------+----------------------+
   | id | group_id | feat                 | value | group_norm           |
   +----+----------+----------------------+-------+----------------------+
   |  1 |  2300555 | shiiiennntaaaahhh/NN |     1 | 0.000479616306954436 |
   |  2 |  2300555 | thx/VBN              |     1 | 0.000479616306954436 |
   |  3 |  2300555 | aku/NN               |     1 | 0.000479616306954436 |
   |  4 |  2300555 | passgae/NN           |     1 | 0.000479616306954436 |
   |  5 |  2300555 | feel/VBP             |     6 |  0.00287769784172662 |
   +----+----------+----------------------+-------+----------------------+
