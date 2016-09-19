.. _fwflag_weighted_lexicon:
==================
--weighted_lexicon
==================
Switch
======

--weighted_lexicon

Description
===========

use with Extraction Action add_lex_table to make weighted lexicon features.

Argument and Default Value
==========================

None

Details
=======

Feature tables created from weighted lexica have _w appended to the second field, e.g., feat$cat_msgsPA_800_cp_w$msgsPA$cnty$16to16


Other Switches
==============

Required Switches:
:doc:`fwflag_add_lex_table` Optional Switches:
None

Example Commands
================
.. code:doc:`fwflag_block`:: python


fwInterface.py :doc:`fwflag_d` paHealth :doc:`fwflag_t` msgsPA :doc:`fwflag_add_lex_table` :doc:`fwflag_l` msgsPA_1000_cp :doc:`fwflag_weighted_lexicon` :doc:`fwflag_c` cnty :doc:`fwflag_word_table` 'feat$1gram$msgsPA$cnty$16to16$0_01'

