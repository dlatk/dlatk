.. _fwflag_topic_lexicon:
===============
--topic_lexicon
===============
Switch
======

--topic_lexicon

Description
===========

Specifies lexicon with appropriately-ranked topics when creating --topic_tagcloud or --make_topic_wordclouds.

Argument and Default Value
==========================

Argument: lexicon name

Details
=======

Generally, these lexica will end with _freq_t50ll-- they are frequency ranked, thresholded at 50, log likelihood.


Other Switches
==============

Required Switches:
:doc:`fwflag_topic_tagcloud` Optional Switches:
None

Example Commands
================
.. code:doc:`fwflag_block`:: python


fwInterface.py :doc:`fwflag_d` paHealth :doc:`fwflag_t` msgsPA :doc:`fwflag_c` cnty :doc:`fwflag_word_table` 'feat$1gram$msgsPA$cnty$16to16$0_01' :doc:`fwflag_group_freq_thresh` 1000 :doc:`fwflag_f` 'feat$cat_msgsPA_800_cp_w$msgsPA$cnty$16to16' :doc:`fwflag_outcome_table` outcome_data_with_controls :doc:`fwflag_outcomes` HIV_rate :doc:`fwflag_correlate` :doc:`fwflag_no_bonf` :doc:`fwflag_p_correction` simes :doc:`fwflag_topic_tagcloud` :doc:`fwflag_make_topic_wordclouds` :doc:`fwflag_topic_lexicon` msgsPA_800_freq_t50ll :doc:`fwflag_output_name` ./output

