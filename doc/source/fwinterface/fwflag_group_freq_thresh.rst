.. _fwflag_group_freq_thresh:
===================
--group_freq_thresh
===================
Switch
======

--group_freq_thresh

Description
===========

Minimum WORD frequency per correl_field to include correl_field in results.

Argument and Default Value
==========================

Argument is an integer number. Unless specified the default value is set based on the `-c` flag: message level gft = 1, user level gft = 1000, county level gft = 40000.

Details
=======

Counts the number of words in each group specified by correl_field. If this count is less than the given group frequency threshold then this group is thrown out. The group is otherwise kept. 


Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Example command correlating county level outcomes with PA twitter data, using Facebook topics and controlling for region
 ./fwInterface.py :doc:`fwflag_d` paHealth :doc:`fwflag_t` msgsPA_2012 :doc:`fwflag_c` cnty :doc:`fwflag_f` 'feat$cat_met_a30_2000_cp_w$msgsPA_2012$cnty$16to16' :doc:`fwflag_outcome_table` outcome_data_with_controls \ 
 :doc:`fwflag_outcomes` diab_perc HIV_rate PAAM_age_adj_mort IM_rate CM_rate food_insec_perc LATHF_perc MV_mort_rate DP_mort_rate UA_perc UC_perc health_care_cost NDDTC_perc \ 
 OPCP_rate HI_income free_lunch_perc HR_rate :doc:`fwflag_topic_tagcloud` :doc:`fwflag_make_topic_wordcloud` :doc:`fwflag_topic_lexicon` met_a30_2000_freq_t50ll :doc:`fwflag_no_bonf` :doc:`fwflag_p_correction` simes \ 
 :doc:`fwflag_output_name` /localdata/paHealth/a19_d6_s4 :doc:`fwflag_controls` 'new_england' 'midatlantic' 'south' 'midwest' 'southwest' 'west' :doc:`fwflag_tagcloud_colorscheme` blue \ 
 :doc:`fwflag_group_freq_thresh` 40000


 # Create a set of 1:doc:`fwflag_grams` from the Primals data at the message level
 # ./fwInterface.py :doc:`fwflag_d` primals :doc:`fwflag_t` primals_new :doc:`fwflag_c` message_id :doc:`fwflag_add_ngrams` :doc:`fwflag_n` 1 :doc:`fwflag_feat_occ_filter` :doc:`fwflag_set_p_occ` 0.001 :doc:`fwflag_group_freq_thresh` 1000