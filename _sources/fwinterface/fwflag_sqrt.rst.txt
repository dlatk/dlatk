.. _fwflag_sqrt:
======
--sqrt
======
Switch
======

--sqrt

Description
===========

A square root transformation of the group norm frequency information.

Argument and Default Value
==========================

Normalizes the group norm frequency information using the following formula:

Details
=======

Normalizes the group norm frequency information using the following formula:



Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` The transformation switches are used during feature extraction and therefore need at least one feature extraction command: :doc:`fwflag_add_ngrams`, :doc:`fwflag_add_lex_table`, etc.
Optional Switches:
None

Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Creates the table: feat$cat_LEX_TABLE$TABLE$GROUP_BY_FIELD$16to4 
./fwInterface.py :doc:`fwflag_d` DATABSE :doc:`fwflag_t` TABLE :doc:`fwflag_c` GROUP_BY_FIELD :doc:`fwflag_add_lex_table` :doc:`fwflag_l` LEX_TABLE  :doc:`fwflag_sqrt` 
 # Creates the table: feat$cat_met_a30_2000_cp_w$primals_new$dp_id$16to4
./fwInterface.py :doc:`fwflag_d` primals :doc:`fwflag_t` primals_new :doc:`fwflag_c` dp_id :doc:`fwflag_group_freq_thresh` 40000 :doc:`fwflag_add_lex_table` :doc:`fwflag_l` met_a30_2000_cp :doc:`fwflag_weighted_lex` :doc:`fwflag_sqrt` 
