.. _fwflag_outcome_with_outcome:
======================
--outcome_with_outcome
======================
Switch
======

--outcome_with_outcome

Description
===========

Adds the outcomes themselves to the list of variables to correlate with the outcomes.

Argument and Default Value
==========================

None, default is false.

Details
=======

When doing correlation analysis (DLA) using :doc:`fwflag_rmatrix`, :doc:`fwflag_correlate` or :doc:`fwflag_tagcloud`, this appends the outcomes to the list of features to be correlated with the outcomes.

This means that the output (rmatrix or other) will have extra lines prefixed with outcome_.


Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f`, :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` Optional Switches:
anything from :doc:`fwflag_correlate` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Correlates LIWC lexical features with age and gender for every user in masterstats_andy_r10k 
 # Also will correlate age and gender over those same users.
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` age gender
 :doc:`fwflag_outcome_with_outcome` :doc:`fwflag_f` 'feat$cat_LIWC2007$messages_en$user_id$16to16' :doc:`fwflag_correlate` 