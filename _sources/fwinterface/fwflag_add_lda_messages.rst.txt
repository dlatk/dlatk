.. _fwflag_add_lda_messages:
==================
--add_lda_messages
==================
Switch
======

--add_lda_messages

Description
===========

Add lda topic version of message table.

Argument and Default Value
==========================

Arguement: LDA_TABLE. There is no default value.

Details
=======

A tokenized message table TOK_TABLE is specified with the :doc:`fwflag_t` switch.  --add_lda_messages then creates the table TOK_TABLE_lda$LDA_TABLE, which has the same structure as TOK_TABLE. 


Other Switches
==============

Required Switches:

* :doc:`fwflag_d` 
* :doc:`fwflag_t` 

Example Commands
================

.. code-block:: python

	# Creates the table twt_20mil_tok_lda$twt_topics
	dlatkInterface.py -d twitterGH -t twt_20mil_tok --add_lda_messages  twt_topics

The table twt_20mil_tok_lda$twt_topics has the same structure as twt_20mil_tok except for the message column. An example of how this column is changed: 

* Message in twt_20mil_tok: ["is", "a", "book"] 
* Message in twt_20mil_tok_lda$twt_topics: [{"index": "0", "term": "book", "doc": "1", "topic_id": "701", "term_id": "5", "message_id": "128679866827677696"}]

In the above command, twt_20mil_tok was created from twt_20mil using :doc:`fwflag_add_tokenized` and the file twt_topics was created addMessageID.py, as in the following two commands:

.. code-block:: python

	dlatkInterface.py -d twitterGH -t twt_20mil -c id --add_tokenized # this creates twt_20mil_tok
	dlatk/addMessageID.py twt_20mil.txt twt_20mil_state.gz > twt_topics

:doc:`../tutorials/tut_lda` 

