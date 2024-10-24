.. _fwflag_add_parses:
============
--add_parses
============
Switch
======

--add_parses

Description
===========

Creates parsed versions of the message table.

Argument and Default Value
==========================

None

Details
=======

This switch creates three tables (for a given table name TABLE): TABLE_const, TABLE_pos, TABLE_dep. Each table contains the same columns and contents as the original table except for the contents of the message column. 

The message column in TABLE_const is now a tree structure corresponding to the grammatical structure of the message. For example:

* original message: Everything is breaking apart.
* parsed message: (ROOT (S (NP (NNP Everything)) (VP (VBZ is) (VP (VBG breaking) (ADVP (RB apart)))) (. .)))

The message column in TABLE_dep is a list of dependencies which provide a representation of grammatical relations between words in a sentence. For example:

* original message: Everything is breaking apart.
* parsed message: ['nsubj(breaking-3, Everything-1)', 'aux(breaking-3, is-2)', 'root(ROOT-0, breaking-3)', 'advmod(breaking-3, apart-4)'] 

The message column in TABLE_pos is a part of speech tagged version of the original message. For example:

* original message: Everything is breaking apart.
* parsed message: Everything/NNP is/VBZ breaking/VBG apart/RB ./.

Note that TABLE_pos is tagged according to the `Penn Treebank Project tags <https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html>`_.

For more details, download the `Part-of-speech tagging annotation style manual <https://www.cis.upenn.edu/~treebank/>`_. 

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 

Example Commands
================

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

