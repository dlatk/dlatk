.. _fwflag_spam_filter:
=============
--spam_filter
=============
Switch
======

--spam_filter threshold

Description
===========

Removes groups (based on :doc:`fwflag_t`) with percentage of spam messages > threshold, writes to new table corptable_nospam with new int column *is_spam*.


Argument and Default Value
==========================

Default threshold = 0.20

Details
=======

Spam words = 'share', 'win', 'check', 'enter', 'products', 'awesome', 'prize', 'sweeps', 'bonus', 'gift'

If any message contains one of the above words it is marked as spam (is_spam = 1, otherwise is_spam = 0). If number of spam messages / total message > threshold then the group is removed from new message table.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t`, :doc:`fwflag_c`

Example Commands
================

.. code-block:: bash
	
	# creates the table msgs_nospam
	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --spam_filter 0.1



