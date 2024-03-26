.. _tut_text_transformations:
=====================================================
Tokenization, Part of Speech Tagging and Segmentation
=====================================================

This is an overview of the different methods for transforming your text. Each method needs the following flags:

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

In every command below we create a new version of the message table which includes all columns from the original table. The only difference is the new table name and the message column. You should consider the following:

* You could be duplicating a lot of unnecessary data
* All of this data could take a long time to write back to MySQL and cause your connection to drop

You should also consider the grouping level (:doc:`../fwinterface/fwflag_c`). Changing this flag in the following commands only changes the chunk size of the reading / writing, which also causes MySQL connections to drop depending on the data size. 

Tokenization
============

Happier Fun Tokenizer
---------------------

Use DLATK's built-in tokenizer Happier Fun Tokenizer, which is an extension of `Happy Fun Tokenizer <http://sentiment.christopherpotts.net/code-data/happyfuntokenizing.py>`_.

* :doc:`../fwinterface/fwflag_add_tokenized`

.. code-block:: bash
	
	# creates the table msgs_tok
	./dlatkInterface.py -d dla_tutorial -t msgs -c message_id --add_tokenized

.. code-block:: mysql 

	mysql> select message from msgs_tok limit 1;
	+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
	| message                                                                                                                                                                                                                                                                                                                |
	+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
	| ["can", "you", "believe", "it", "?", "?", "my", "mom", "wouln't", "let", "me", "go", "out", "on", "my", "b'day", "...", "i", "was", "really", "really", "mad", "at", "her", ".", "still", "am", ".", "but", "i", "got", "more", "presents", "from", "my", "friends", "this", "year", ".", "so", "thats", "great", "."] |
	+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

TweetNLP Tokenizer
------------------

Use Carnegie Mellon University's `TweetNLP <http://www.cs.cmu.edu/~ark/TweetNLP/>`_ tokenizer.

* :doc:`../fwinterface/fwflag_add_tweettok`

.. code-block:: bash
	
	# creates the table msgs_tweettok 
	./dlatkInterface.py -d dla_tutorial -t msgs -c message_id --add_tweettok

.. code-block:: mysql 

	mysql> select message from msgs_tweettok limit 1;
	+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
	| message                                                                                                                                                                                                                                                                                                            |
	+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
	| ["can", "you", "believe", "it", "??", "my", "mom", "wouln't", "let", "me", "go", "out", "on", "my", "b'day", "...", "i", "was", "really", "really", "mad", "at", "her", ".", "still", "am", ".", "but", "i", "got", "more", "presents", "from", "my", "friends", "this", "year", ".", "so", "thats", "great", "."] |
	+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Sentence Tokenization
---------------------

* :doc:`../fwinterface/fwflag_add_sent_tokenized`

.. code-block:: bash
	
	# creates the table msgs_stokes
	./dlatkInterface.py -d dla_tutorial -t msgs -c message_id --add_sent_tokenized

.. code-block:: mysql 

	mysql> select message_id, message from msgs_stoks limit 1;
	+------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
	| message_id | message                                                                                                                                                                                      |
	+------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
	|          1 | ["can you believe it??", "my mom wouln't let me go out on my b'day...i was really really mad at her.", "still am.", "but i got more presents from my friends this year.", "so thats great."] |
	+------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Or you can save each sentence as it's own row in MySQL:

* :doc:`../fwinterface/fwflag_add_sent_per_row`

.. code-block:: bash
	
	# creates the table msgs_sent
	./dlatkInterface.py -d dla_tutorial -t msgs -c message_id --add_sent_per_row

.. code-block:: mysql 

	mysql> select message_id, message from msgs_sent limit 5;
	+------------+----------------------------------------------------------------------------+
	| message_id | message                                                                    |
	+------------+----------------------------------------------------------------------------+
	| 1_01       | can you believe it??                                                       |
	| 1_02       | my mom wouln't let me go out on my b'day...i was really really mad at her. |
	| 1_03       | still am.                                                                  |
	| 1_04       | but i got more presents from my friends this year.                         |
	| 1_05       | so thats great.                                                            |
	+------------+----------------------------------------------------------------------------+

Part of Speech Tagging
======================

Stanford Parser
---------------

Use the Stanford Parser to create three tables:

* msgs_const - a tree structure corresponding to the grammatical structure of the message
* msgs_pos - a part of speech tagged version of the original message
* msgs_dep - a list of dependencies which provide a representation of grammatical relations between words in a sentence. 

Use the flag:

* :doc:`../fwinterface/fwflag_add_parses`

.. code-block:: bash
	
	# creates the table msgs_const, msgs_pos, msgs_dep
	./dlatkInterface.py -d dla_tutorial -t msgs -c message_id --add_parses

.. code-block:: mysql 

	mysql> select message from msgs_const limit 1;
	+-------------------------------------------------------------------------------------------------+
	| message                                                                                         |
	+-------------------------------------------------------------------------------------------------+
	| (ROOT (S (VP (VB urlLink) (NP (DT The) (NNP Obligatory) (NNP Field) (NNP Shot) (NN urlLink))))) |
	+-------------------------------------------------------------------------------------------------+

	mysql> select message from msgs_pos limit 1;
	+----------------------------------------------------------------+
	| message                                                        |
	+----------------------------------------------------------------+
	| urlLink/VB The/DT Obligatory/NNP Field/NNP Shot/NNP urlLink/NN |
	+----------------------------------------------------------------+

	mysql> select message from msgs_dep limit 1;
	+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
	| message                                                                                                                                                              |
	+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
	| ['root(ROOT-0, urlLink-1)', 'det(urlLink-6, The-2)', 'nn(urlLink-6, Obligatory-3)', 'nn(urlLink-6, Field-4)', 'nn(urlLink-6, Shot-5)', 'dobj(urlLink-1, urlLink-6)'] |
	+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Note that msgs_pos is tagged according to the `Penn Treebank Project tags <https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html>`_.

TweetNLP Part of Speech Tags
----------------------------

Use Carnegie Mellon University's `TweetNLP <http://www.cs.cmu.edu/~ark/TweetNLP/>`_  part of speech tagger.

* :doc:`../fwinterface/fwflag_add_tweetpos`

.. code-block:: bash
	
	# creates the table msgs_tweetpos
	./dlatkInterface.py -d dla_tutorial -t msgs -c message_id --add_tweetpos

.. code-block:: mysql 

	mysql> select message from msgs_tweetpos limit 1;
	+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
	| message                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
	+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
	| {"probs": ["0.9990", "0.9993", "0.9999", "0.9853", "0.9934", "0.9958", "0.9813", "0.9890", "0.9999", "0.9994", "0.9973", "0.7924", "0.9962", "0.9963", "0.9934", "0.9776", "0.9931", "0.9997", "0.9997", "0.9997", "0.9505", "0.9997", "0.8819", "0.9984", "0.9925", "0.9268", "0.9984", "0.9964", "0.9957", "0.9996", "0.6084", "0.5645", "0.9990", "0.9986", "0.9735", "0.9791", "0.9904", "0.9991", "0.5527", "0.9695", "0.9981", "0.9985"], "tags": ["V", "O", "V", "O", ",", "D", "N", "V", "V", "O", "V", "T", "P", "D", "N", ",", "O", "V", "R", "R", "A", "P", "O", ",", "R", "V", ",", "&", "O", "V", "A", "V", "P", "D", "N", "D", "N", ",", "P", "L", "A", ","], "tokens": ["can", "you", "believe", "it", "??", "my", "mom", "wouln't", "let", "me", "go", "out", "on", "my", "b'day", "...", "i", "was", "really", "really", "mad", "at", "her", ".", "still", "am", ".", "but", "i", "got", "more", "presents", "from", "my", "friends", "this", "year", ".", "so", "thats", "great", "."], "original": "can you believe it?? my mom wouln't let me go out on my b'day...i was really really mad at her. still am. but i got more presents from my friends this year. so thats great."} |
	+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


Segmentation with the Stanford Segmenter
========================================

For Chinese text you can use the `Stanford segmenter <https://nlp.stanford.edu/software/segmenter.shtml>`_. Use the :doc:`../fwinterface/fwflag_segmentation_model` flag to change models: ctb (default, Penn Chinese Treebank) or pku (Beijing University).

* :doc:`../fwinterface/fwflag_add_segmented`
* :doc:`../fwinterface/fwflag_segmentation_model`

.. code-block:: bash
	
	# creates the table msgs_seg via the Penn Chinese Treebank
	./dlatkInterface.py -d dla_tutorial -t msgs -c message_id --add_segmented

.. code-block:: mysql 

	mysql> select message from msgs_ch limit 1;
	+------------------------------------------------------------------------------+
	| message                                                                      |
	+------------------------------------------------------------------------------+
	| [神马]欧洲站夏季女装雪纺短袖长裤女士运动时尚休闲套装女夏装2014新款  http://t.cn/RvCypCj |
	+------------------------------------------------------------------------------+

	mysql> select message from msgs_ch_seg limit 1;
	+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
	| message                                                                                                                                                                                                                                                                               |
	+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
	| ["[\u795e\u9a6c]", "\u6b27\u6d32", "\u7ad9", "\u590f\u5b63", "\u5973\u88c5", "\u96ea\u7eba", "\u77ed\u8896", "\u957f\u88e4", "\u5973\u58eb", "\u8fd0\u52a8", "\u65f6\u5c1a", "\u4f11\u95f2", "\u5957\u88c5", "\u5973", "\u590f\u88c5", "2014", "\u65b0\u6b3e", "http://t.cn/RvCypCj"] |
	+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

