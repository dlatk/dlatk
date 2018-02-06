.. _fwflag_logistic_reg:
==============
--logistic_reg
==============
Switch
======

--logistic_reg

Description
===========

Use logistic regression instead of linear regression. This is better for binary outcomes.

Argument and Default Value
==========================

None

Details
=======

Note: you cannot compare coefficients in Logistic regression. See this article for more info. You can compare p values, though. See :doc:`fwflag_correlate` for more info on correlation.


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`
* :doc:`fwflag_f`
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes`

Optional Switches:

* :doc:`fwflag_interaction_ddla`
* :doc:`fwflag_correlate`
* :doc:`fwflag_rmatrix`
* :doc:`fwflag_tagcloud`
* :doc:`fwflag_topic_tagcloud`
* :doc:`fwflag_barplot`
* :doc:`fwflag_feat_correl_filter`
* :doc:`fwflag_make_wordclouds`
* :doc:`fwflag_make_topic_wordclouds`

Example Commands
================
.. code:doc:`fwflag_block`:: python
