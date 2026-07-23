.. _fwflag_loessplot:
===========
--loessplot
===========
Switch
======

--loessplot FEAT_1 ... FEAT_N

Description
===========

Output loess plots of the given features.

Argument and Default Value
==========================

Space separated list of feature names.

Details
=======

LOESS and LOWESS (locally weighted scatterplot smoothing) are two strongly related non:doc:`fwflag_parametric` regression methods that combine multiple regression models in a k:doc:`fwflag_nearest-neighbor-based` meta:doc:`fwflag_model`. "LOESS" is a later generalization of LOWESS; although it is not a true initialism, it may be understood as standing for "LOcal regrESSion".


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_corpdb`
* :doc:`fwflag_f`, :doc:`fwflag_feat_table`
* :doc:`fwflag_outcomes` :doc:`fwflag_outcome_table`

Optional Switches:

* :doc:`fwflag_output_dir`
* :doc:`fwflag_spearman`
* :doc:`fwflag_group_freq_thresh`
* :doc:`fwflag_no_bonferroni`
* :doc:`fwflag_p_correction`
* :doc:`fwflag_blacklist`
* :doc:`fwflag_whitelist`
* :doc:`fwflag_show_feat_freqs`
* :doc:`fwflag_not_show_feat_freqs`
* :doc:`fwflag_output_dir`
* :doc:`fwflag_output_name`
* :doc:`fwflag_topic_lexicon`

Example Commands
================
.. code-block:: doc:`fwflag_block`:: python
