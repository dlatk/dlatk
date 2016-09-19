.. _fwflag_p_correction:
==============
--p_correction
==============
Switch
======

--p_correction

Description
===========

Specifies a p-value correction method (for multiple comparisons) in correlation other than Bonferroni (which is turned off with --no_bonferroni)

Argument and Default Value
==========================

Argument: method to use

Details
=======

Possible values include: simes, holm, hochberg, hommel, bonferroni, BH, BY, fdr, none

simes is built into featureWorker; anything else calls R's stats module, specifically the p_adjust command


Other Switches
==============

Required Switches:
:doc:`fwflag_no_bonferroni` :doc:`fwflag_correlate` Optional Switches:
None

Example Commands
================
.. code:doc:`fwflag_block`:: python


