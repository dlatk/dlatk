.. _fwflag_deduplicate:
=============
--deduplicate
=============
Switch
======

--deduplicate

Description
===========

Removes duplicate tweets within :doc:`fwflag_t` grouping, writes to new table corptable_dedup. Not to be run at the message level.


Argument and Default Value
==========================

None

Details
=======

Takes a mysql message table and removes all duplicate messages within a given user. Duplicate tweets = any tweets with the same first 6 tokens (no usernames, no url, no hashtags, no smileys, no punctuation, etc.). Writes to new message table with _dedup appended to end of name. For example, the following two tweets would be considered duplicates despite not being identical:

* this is a tweet with a url http://t.co/qT62KOdzeW http://t.co/MsZ2vHJ4H0
* this is a tweet with a url http://t.co/W6m3uPju4P

Written by Daniel Preotiuc, original code found `here <https://github.com/danielpreotiuc/trendminer-clustering/blob/master/python/dedup.py>`_.

The :doc:`fwflag_clean_messages` flag will remove urls (and replace with <URL>) and @ mentions (and replace with <USER>).

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 

Optional Switches:

* :doc:`fwflag_clean_messages`

Example Commands
================

Remove duplicate tweets while cleaning URLs and @mentions:

.. code-block:: bash

	# creates the table msgs_dedup
	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --deduplicate --clean_messages

