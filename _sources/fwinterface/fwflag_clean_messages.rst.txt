.. _fwflag_clean_messages:
================
--clean_messages
================
Switch
======

--clean_messages

Description
===========

When used alone it replaces URLs with <URL> and @mentions with <USER>.

When used with:

* :doc:`fwflag_deduplicate`: it replaces URLs with <URL> and @mentions with <USER> but also removed duplicate tweets.
* :doc:`fwflag_language_filter`: it removed urls and @mentions before applying the language filter but not removed from the resulting message table.


Argument and Default Value
==========================

None

Details
=======

When used alone it will create a new table whose name is taken from the :doc:`fwflag_t` flag and appends "_an".

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 

Optional Switches:

* :doc:`fwflag_deduplicate`
* :doc:`fwflag_language_filter`

Example Commands
================

Clean URLs and @mentions:

.. code-block:: bash
	
	# creates the table msgs_an
	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --clean_messages

Clean URLs and @mentions while lanugage filtering:

.. code-block:: bash

	# creates the table msgs_en
	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --language_filter en --clean_messages

Clean URLs and @mentions while deduplicating:

.. code-block:: bash

	# creates the table msgs_dedup
	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --deduplicate --clean_messages

