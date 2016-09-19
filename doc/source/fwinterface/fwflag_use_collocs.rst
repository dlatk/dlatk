.. _fwflag_use_collocs:
=============
--use_collocs
=============
Switch
======

--use_collocs

Description
===========

Use a set of collocations to extract n grams.

Argument and Default Value
==========================

Use this option to extract features using a collocation table (--colloc_table), or to modify a feature table that was extracted using collocations.  The collocation table holds the multigrams that should be considered together.  All words that aren’t part of the predefined list of collocations will be counted as 1grams.

Details
=======

Use this option to extract features using a collocation table (:doc:`fwflag_colloc_table`), or to modify a feature table that was extracted using collocations.  The collocation table holds the multigrams that should be considered together.  All words that aren’t part of the predefined list of collocations will be counted as 1grams.  

Note: :doc:`fwflag_colloc_table` is assumed to have columns ‘feat’

Note: The preferred collocation table as of June 2015 is ufeat$pmi$fb22_messagesEn$lnpmi0_15


Other Switches
==============

Required Switches:
None
Optional Switches:
:doc:`fwflag_colloc_table` <TABLENAME>
:doc:`fwflag_include_sub_collocs` :doc:`fwflag_feature_type_name` <STRING>

Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Extract and filter in one command
 ./fwInterface.py :doc:`fwflag_d` fb22 :doc:`fwflag_t` msgsEn_r5k :doc:`fwflag_c` user_id :doc:`fwflag_add_ngrams` :doc:`fwflag_use_collocs` :doc:`fwflag_colloc_table` 'ufeat$pmi$fb22_messagesEn$lnpmi0_15' :doc:`fwflag_feat_occ_filter` :doc:`fwflag_set_p_occ` 0.05


 # Add a filter to a table that was generated using collocs, (requires specifying the word table for group_frequency calculation)
 ./fwInterface.py :doc:`fwflag_d` fb22 :doc:`fwflag_t` msgsEn_r5k :doc:`fwflag_c` user_id :doc:`fwflag_f` ’feat$colloc$msgsEn_r5k$user_id$16to16’ :doc:`fwflag_word_table` ’feat$colloc$msgsEn_r5k$user_id$16to16’ :doc:`fwflag_feat_occ_filter` :doc:`fwflag_set_p_occ` 0.05
Example outputs: 
feat$colloc$msgsEn_r5k$user_id$16to16
feat$colloc$msgsEn_r5k$user_id$16to16$0_05
Off:doc:`fwflag_label` use: only extract 1:doc:`fwflag_grams` that appear in the lex table ANEW:
fwInterface.py :doc:`fwflag_d` fbtrust :doc:`fwflag_t` messagesEn :doc:`fwflag_c` user_id :doc:`fwflag_add_ngrams` :doc:`fwflag_use_collocs` :doc:`fwflag_colloc_table` ANEW :doc:`fwflag_colloc_column` term :doc:`fwflag_feature_type_name` ANEWterms
