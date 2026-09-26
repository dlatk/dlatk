.. _fwflag_add_tokenized:
===============
--add_tokenized
===============
Switch
======

--add_tokenized

Description
===========

Creates a tokenized version of the message table.

Argument and Default Value
==========================

None

Details
=======

This will create a table called TABLE_tok (where TABLE is specified by :doc:`fwflag_t`) in the database specified by :doc:`fwflag_d`. The message column in this new table is a list of tokens. It uses DLATK's built-in tokenizer Happier Fun Tokenizer, which is an extension of `Happy Fun Tokenizer <http://sentiment.christopherpotts.net/code-data/happyfuntokenizing.py>`_.

If your message is:

.. code-block:: bash

	"Mom said she's gonna think about getting a truck."

the same row in the tokenized table will look like this:

.. code-block:: bash

	["mom", "said", "she's", "gonna", "think", "about", "getting", "a", "truck", "."]

To use the tokenized table in standalone scripts, simply do JSON.load(message).


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 


Example Commands
================

.. code-block:: bash

	# Creates the tables: msgs_tok
	dlatkInterface.py -d dla_tutorial -t msgs -c message_id --add_tokenized

.. code-block:: mysql 

	mysql> select message from msgs_tok limit 1;
	+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
	| message                                                                                                                                                                                                                                                                                                                |
	+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
	| ["can", "you", "believe", "it", "?", "?", "my", "mom", "wouln't", "let", "me", "go", "out", "on", "my", "b'day", "...", "i", "was", "really", "really", "mad", "at", "her", ".", "still", "am", ".", "but", "i", "got", "more", "presents", "from", "my", "friends", "this", "year", ".", "so", "thats", "great", "."] |
	+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
