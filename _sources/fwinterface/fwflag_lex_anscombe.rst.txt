.. _fwflag_lex_anscombe:
==============
--lex_anscombe
==============
Switch
======

--lex_anscombe

Description
===========

Apply a transformation to the group norm before

Argument and Default Value
==========================

extracting from boolean words (or other types of transformations for that matter)
If the lexicon was build to measure if a word W appears in a group (vs. what percentage of that group's word use was made up of this word W)
Before, you would simply run the extraction, not being able to do that.
For more information on the anscombe transformation see documentation for --anscombe

Details
=======

extracting from boolean words (or other types of transformations for that matter)
If the lexicon was build to measure if a word W appears in a group (vs. what percentage of that group's word use was made up of this word W)
Before, you would simply run the extraction, not being able to do that.
For more information on the anscombe transformation see documentation for :doc:`fwflag_anscombe`

Other Switches
==============

Required Switches:

* :doc:`fwflag_add_lex_table`
* :doc:`fwflag_l` <LEX_TABLE>

Optional Switches:

None

Example Commands
================
.. code-block:: doc:`fwflag_block`:: python


 # Correlates 1grams with age for every user
 ./fwInterface.py :doc:`fwflag_d` DB :doc:`fwflag_t` msgs_table :doc:`fwflag_c` groupId :doc:`fwflag_add_lex_table` :doc:`fwflag_l` myLexTable :doc:`fwflag_lex_anscombe`
