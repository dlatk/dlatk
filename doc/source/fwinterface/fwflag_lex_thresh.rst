.. _fwflag_lex_thresh:
============
--lex_thresh
============
Switch
======

--lex_thresh THRESH

Description
===========

Used with --add_topiclex_from_topicfile and --topic_lex_method csv_lik: keeps only the words whose weight exceeds THRESH times the weight of their topic's top word.

Argument and Default Value
==========================

THRESH - a fraction, e.g. 0.5. Default: keep all words.

Other Switches
==============

Required Switches:

* --add_topiclex_from_topicfile
* --topic_file
* :doc:`fwflag_topic_lexicon`

Example Commands
================

.. code-block:: bash

	# make a weighted topic lexicon, keeping words with at least half their topic's top weight
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_topiclex_from_topicfile --topic_file topics.csv --topic_lexicon my_topics_cp --topic_lex_method csv_lik --lex_thresh 0.5
