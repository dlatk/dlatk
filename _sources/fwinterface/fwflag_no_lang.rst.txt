.. _fwflag_no_lang:
=========
--no_lang
=========
Switch
======

--no_lang

Description
===========

Runs --combo_test_regression or --combo_test_classifiers without language features.

Argument and Default Value
==========================

None

Details
=======


Other Switches
==============

Required Switches:
:doc:`fwflag_combo_test_regression` OR
:doc:`fwflag_combo_test_classifiers` Optional Switches:
None

Example Commands
================
.. code:doc:`fwflag_block`:: python


 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01' 
 :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` age :doc:`fwflag_combo_test_regression` :doc:`fwflag_model` ridgecv :doc:`fwflag_folds` 10
 :doc:`fwflag_no_lang` 