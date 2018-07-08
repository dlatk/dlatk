.. _fwflag_log:
=====
--log
=====
Switch
======

--log

Description
===========

A log transformation of the group norm frequency information.

Details
=======

Normalizes the group norm frequency information using the following formula: `ln(x+1)`.



Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`

The transformation switches are used during feature extraction and therefore need at least one feature extraction command:

* :doc:`fwflag_add_ngrams`, :doc:`fwflag_add_lex_table`, etc.



Example Commands
================

.. code-block:: bash

	# produces the table feat$1gram$msgs$user_id$16to3
   	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_ngrams -n 1 --log

	# produces the table feat$cat_met_a30_2000_cp_w$msgs$user_id$16to3$1gra
	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_lex_table -l met_a30_2000_cp --weighted_lex --log

