.. _fwflag_spearman:
==========
--spearman
==========
Switch
======

--spearman

Description
===========

Returns Spearman correlation instead of Pearson R

Argument and Default Value
==========================

None

Details
=======


Other Switches
==============

Required Switches:
:doc:`fwflag_correlate` Optional Switches:
None

Example Commands
================
.. code:doc:`fwflag_block`:: python


fwInterface.py :doc:`fwflag_d` tester7 :doc:`fwflag_t` statuses_er1 :doc:`fwflag_c` study_code :doc:`fwflag_f` 'feat$cat_statuses_er1_cp_w$statuses_er1$study_code$16to16$0_05' :doc:`fwflag_group_freq_thresh` 500 :doc:`fwflag_outcome_table` EDinfo11_11_14 :doc:`fwflag_outcomes` 6moPostCount :doc:`fwflag_outcome_controls` sex_int isBlack isWhite isHispanic ageTercile0 ageTercile1 ageTercile2 :doc:`fwflag_correlate` :doc:`fwflag_spearman` :doc:`fwflag_rmatrix` :doc:`fwflag_topic_tagcloud` :doc:`fwflag_topic_lexicon` statuses_er1_freq_t50ll :doc:`fwflag_make_topic_wordclouds` :doc:`fwflag_no_bonferroni` :doc:`fwflag_p_correction` simes :doc:`fwflag_sort` :doc:`fwflag_tagcloud_colorscheme` blue :doc:`fwflag_output_name` ./output

