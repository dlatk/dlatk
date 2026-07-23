.. _fwflag_add_sent_per_row:
==================
--add_sent_per_row
==================
Switch
======

--add_sent_per_row

Description
===========

Use the `NLTK <www.nltk.org>`_ Punkt sentence tokenizer to create a sentence tokenized version of the message table. 

Argument and Default Value
==========================

None

Details
=======

Each row in the message table is sentence tokenized and each sentence is written to a single row in MySQL. The message id column of the tokenized sentences corresponds to the original message id with a number added to the end for each sentence. 


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 


Example Commands
================

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