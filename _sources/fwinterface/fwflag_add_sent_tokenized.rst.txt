.. _fwflag_add_sent_tokenized:
====================
--add_sent_tokenized
====================
Switch
======

--add_sent_tokenized

Description
===========

Use the `NLTK <www.nltk.org>`_ Punkt sentence tokenizer to create a sentence tokenized version of the message table. 

Argument and Default Value
==========================

None

Details
=======

Each row in the message table is sentence tokenized and written as a list to a single row in MySQL. Use :doc:`_fwflag_add_sent_per_row` to write each sentence to its own row. 

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 


Example Commands
================

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
