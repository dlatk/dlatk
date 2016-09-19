.. _fwflag_output_name:
=============
--output_name
=============
Switch
======

--output_name FILENAME

Description
===========

Overrides the default filename for output.

Argument and Default Value
==========================

String for output file name.

Details
=======


Other Switches
==============

This is optional for many switches. 

Optional Switches
:doc:`fwflag_cca` :doc:`fwflag_rmatrix` :doc:`fwflag_mediation` :doc:`fwflag_combo_rmatrix`? :doc:`fwflag_loessplot` :doc:`fwflag_ddlaFiles`? :doc:`fwflag_combo_test_regression` :doc:`fwflag_csv` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Example command correlating county level outcomes with PA twitter data, using Facebook topics and controlling for region
 ./fwInterface.py :doc:`fwflag_d` paHealth :doc:`fwflag_t` msgsPA_2012 :doc:`fwflag_c` cnty :doc:`fwflag_f` 'feat$cat_met_a30_2000_cp_w$msgsPA_2012$cnty$16to16' :doc:`fwflag_outcome_table` outcome_data_with_controls \ 
 :doc:`fwflag_outcomes` diab_perc HIV_rate PAAM_age_adj_mort IM_rate CM_rate food_insec_perc LATHF_perc MV_mort_rate DP_mort_rate UA_perc UC_perc health_care_cost NDDTC_perc \ 
 OPCP_rate HI_income free_lunch_perc HR_rate :doc:`fwflag_topic_tagcloud` :doc:`fwflag_make_topic_wordcloud` :doc:`fwflag_topic_lexicon` met_a30_2000_freq_t50ll :doc:`fwflag_no_bonf` :doc:`fwflag_p_correction` simes \ 
 :doc:`fwflag_output_name` /localdata/paHealth/a19_d6_s4 :doc:`fwflag_controls` 'new_england' 'midatlantic' 'south' 'midwest' 'southwest' 'west' :doc:`fwflag_tagcloud_colorscheme` blue \ 
 :doc:`fwflag_group_freq_thresh` 40000