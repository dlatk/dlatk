.. _fwflag_weighted_lexicon:
==================
--weighted_lexicon
==================
Switch
======

--weighted_lexicon

Description
===========

Use with add_lex_table to make weighted lexicon features.

Argument and Default Value
==========================

None

Details
=======

Feature tables created from weighted lexica have _w appended to the second field, e.g., feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16


Other Switches
==============

Required Switches:

* :doc:`fwflag_add_lex_table` 

Optional Switches:

* None

Example Commands
================

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_lex_table -l met_a30_2000 --weighted_lexicon  --word_table 'feat$1gram$msgs$user_id$16to16$0_01'

