.. _fwflag_scatterplot:
=============
--scatterplot
=============
Switch
======

--scatterplot

Description
===========

Creates a scatter plot for each pair of given variables.

Argument and Default Value
==========================

None

Details
=======

If no feature table is given then a scatter plot is created for each pair of variables listed in :doc:`fwflag_outcomes`. 

If a feature table is given then a scatter plot is created for every pair of variables (feat, outcome), where feat is specified by :doc:`fwflag_feature_names`? and outcome is specified by :doc:`fwflag_outcomes`. 


Other Switches
==============

Required Switches:
:doc:`fwflag_outcomes` :doc:`fwflag_outcome_table` Optional Switches:
:doc:`fwflag_f` :doc:`fwflag_feature_names`? 
Example Commands
================
.. code:doc:`fwflag_block`:: python
