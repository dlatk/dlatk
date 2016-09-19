.. _fwflag_outcome_with_outcome_only:
===========================
--outcome_with_outcome_only
===========================
Switch
======

--outcome_with_outcome_only

Description
===========

Correlate a list of outcomes with each other

Argument and Default Value
==========================

None, default is false.

Details
=======

Similar to :doc:`fwflag_outcome_with_outcome` except no language features are considered. 

If a feature table is given then :doc:`fwflag_group_freq_thresh` will be applied, otherwise we consider all users in the outcome table.


Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` Optional Switches:
:doc:`fwflag_t`, :doc:`fwflag_f` :doc:`fwflag_group_freq_thresh` anything from :doc:`fwflag_correlate` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Correlates age and gender for every user in masterstats_andy_r10k who wrote at least 100 words 
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` age gender
 :doc:`fwflag_outcome_with_outcome_only` :doc:`fwflag_f` 'feat$cat_LIWC2007$messages_en$user_id$16to16' :doc:`fwflag_correlate` :doc:`fwflag_group_freq_thresh` 100

 # Correlates age and gender for every user in masterstats_andy_r10k  
 ~/fwInterface.py :doc:`fwflag_d` fb20  :doc:`fwflag_c` user_id :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` age gender
 :doc:`fwflag_outcome_with_outcome_only` 
