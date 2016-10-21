.. _fwflag_add_parses:
============
--add_parses
============
Switch
======

--add_parses

Description
===========

Creates parsed versions of the message table.

Argument and Default Value
==========================

None

Details
=======

This switch creates three tables (for a given table name TABLE): TABLE_const, TABLE_pos, TABLE_dep. Each table contains the same columns and contents as the original table except for the contents of the message column. The message column in TABLE_const is now a tree structure corresponding to the grammatical structure of the message. For example:
original message: Everything is breaking apart.
parsed message: (ROOT (S (NP (NNP Everything)) (VP (VBZ is) (VP (VBG breaking) (ADVP (RB apart)))) (. .)))
The message column in TABLE_dep is a list of dependencies which provide a representation of grammatical relations between words in a sentence. For example:
original message: Everything is breaking apart.
parsed message: ['nsubj(breaking:doc:`fwflag_3`, Everything:doc:`fwflag_1`)', 'aux(breaking:doc:`fwflag_3`, is:doc:`fwflag_2`)', 'root(ROOT:doc:`fwflag_0`, breaking:doc:`fwflag_3`)', 'advmod(breaking:doc:`fwflag_3`, apart:doc:`fwflag_4`)'] The message column in TABLE_pos is a part of speech tagged version of the original message. For example:
original message: Everything is breaking apart.
parsed message: Everything/NNP is/VBZ breaking/VBG apart/RB ./.
Note: TABLE_pos is tagged according to the Penn Treebank Project tags: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
For more details, download the Part:doc:`fwflag_of-speech` tagging annotation style manual from https://www.cis.upenn.edu/~treebank/

Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` Optional Switches:
None

Example Commands
================
.. code:doc:`fwflag_block`:: python


 # General form
 # Creates the tables: TABLE_const, TABLE_dep, TABLE_pos
 ./fwInterface.py :doc:`fwflag_d` DATABASE :doc:`fwflag_t` TABLE :doc:`fwflag_c` GROUP_BY_FIELD :doc:`fwflag_add_parses` 
 # Creates the tables: primals_new_const, primals_new_dep, primals_new_pos
 ./fwInterface.py :doc:`fwflag_d` primals :doc:`fwflag_t` primals_new :doc:`fwflag_c` message_id :doc:`fwflag_add_parses` 