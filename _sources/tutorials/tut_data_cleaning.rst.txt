.. _tut_data_cleaning:
=============
Data Cleaning
=============

This is an overview of the different methods for cleaning text data. Each method needs the following flags:

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

Language Filtering
------------------

Uses the `langid <https://github.com/saffsd/langid.py>`_ python package. For each message it will assign a confidence and keep the message if the confidence is over 0.80. By default this will lowercase your messages before running through langid. 

This will create a new table whose name is taken from the -t flag and appends "_en". 

* :doc:`../fwinterface/fwflag_language_filter`

.. code-block:: bash

	# creates the table msgs_en
	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --language_filter en

Adding the :doc:`../fwinterface/fwflag_clean_messages` flag will remove URLs and @-mentions before applying the language filter, which improves the language classification

.. code-block:: bash

	# creates the table msgs_en
	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --language_filter en --clean_messages

Deduplicating
-------------

Removes tweets which contains the same first 6 tokens as another message within the same group (-c) (therefore each group id should have more than one row in your MySQL table). This will create a new table whose name is taken from the -t flag and appends "_dedup".

* :doc:`../fwinterface/fwflag_deduplicate`

.. code-block:: bash

	# creates the table msgs_dedup
	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --deduplicate

To also replace URLs (with <URL>) and @-mentions (with <USER>) add the :doc:`../fwinterface/fwflag_clean_messages` flag (this helps the language classification task):

.. code-block:: bash

	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --deduplicate --clean_messages

Removing URLs and @-mentions
----------------------------

When using :doc:`../fwinterface/fwflag_clean_messages` alone it will create a new table whose name is taken from the :doc:`../fwinterface/fwflag_t` flag and appends "_an".

* :doc:`../fwinterface/fwflag_clean_messages`

.. code-block:: bash
	
	# creates the table msgs_an
	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --clean_messages

Spam Filtering
--------------

If any message contains one of the spam words it is marked as spam (is_spam = 1, otherwise is_spam = 0). If number of spam messages / total message > THRESHOLD then user is removed from new message table.

Spam words = 'share', 'win', 'check', 'enter', 'products', 'awesome', 'prize', 'sweeps', 'bonus', 'gift'

This will create a new table whose name is taken from the -t flag and appends "_nospam".

* :doc:`../fwinterface/fwflag_spam_filter`

.. code-block:: bash
	
	# creates the table msgs_nospam
	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --spam_filter 0.1

Real World Example
------------------

Using both the language filter and the deduplication filters typically yeild nice results (both with the :doc:`../fwinterface/fwflag_clean_messages` flag):

.. code-block:: bash

	# creates the table msgs_en
	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --language_filter en --clean_messages

	# creates the table msgs_en_dedup
	./dlatkInterface.py -d dla_tutorial -t msgs_en -c user_id --deduplicate --clean_messages

