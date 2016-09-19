.. _fwflag_outcome_fields:
================
--outcome_fields
================
Switch
======

--outcome_fields, --outcomes

Description
===========

Fields to compare with.

Argument and Default Value
==========================

A list of column names, separated by a space. There is no default value.

Details
=======

Other switches

Required switches
:doc:`fwflag_outcome_table` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Example command correlating county level outcomes with PA twitter data, using Facebook topics and controlling for region
 ./fwInterface.py :doc:`fwflag_d` paHealth :doc:`fwflag_t` msgsPA_2012 :doc:`fwflag_c` cnty :doc:`fwflag_f` 'feat$cat_met_a30_2000_cp_w$msgsPA_2012$cnty$16to16' :doc:`fwflag_outcome_table` outcome_data_with_controls \ 
 :doc:`fwflag_outcomes` diab_perc HIV_rate PAAM_age_adj_mort IM_rate CM_rate food_insec_perc LATHF_perc MV_mort_rate DP_mort_rate UA_perc UC_perc health_care_cost NDDTC_perc \ 
 OPCP_rate HI_income free_lunch_perc HR_rate :doc:`fwflag_topic_tagcloud` :doc:`fwflag_make_topic_wordcloud` :doc:`fwflag_topic_lexicon` met_a30_2000_freq_t50ll :doc:`fwflag_no_bonf` :doc:`fwflag_p_correction` simes \ 
 :doc:`fwflag_output_name` /localdata/paHealth/a19_d6_s4 :doc:`fwflag_controls` 'new_england' 'midatlantic' 'south' 'midwest' 'southwest' 'west' :doc:`fwflag_tagcloud_colorscheme` blue


 # Example using mediation analysis
 ./fwInterface.py :doc:`fwflag_d` twitterGH :doc:`fwflag_t` messages_en :doc:`fwflag_c` cty_id :doc:`fwflag_f` 'feat$cat_met_a30_2000_cp_w$messages_en$cty_id$16to16' :doc:`fwflag_mediation` :doc:`fwflag_path_start` '1051' \ 
 :doc:`fwflag_outcomes` 'hsgradHC03_VC93ACS3yr$10' :doc:`fwflag_mediators`  'hsgradHC03_VC93ACS3yr$10' :doc:`fwflag_outcome_table` nejm_intersect_small50k