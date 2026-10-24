.. _fwflag_add_tweettok:
==============
--add_tweettok
==============
Switch
======

--add_tweettok

Description
===========
Use Carnegie Mellon University's `TweetNLP <http://www.cs.cmu.edu/~ark/TweetNLP/>`_ tokenizer to create a tokenized version of the message table.

Argument and Default Value
==========================

None

Details
=======

This will create a table called TABLE_tweettok (where TABLE is specified by :doc:`fwflag_t`) in the database specified by :doc:`fwflag_d`. The message column in this new table is a list of tokens. 

Example on one message

Original message:

.. code-block:: bash

	"@antijokeapple: What do you call a Bee who is having a bad hair day? A Frisbee." Hahah. 

Tokenized message:

.. code-block:: bash

	["\"", "@antijokeapple", ":", "What", "do", "you", "call", "a", "Bee", "who", "is", "having", "a", "bad", "hair", "day", "?", "A", "Frisbee", ".", "\"", "Hahah", "."]

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 


Example Commands
================

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
