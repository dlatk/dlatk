.. _fwflag_outcomes:
==========
--outcomes
==========
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

Required switches

* :doc:`fwflag_outcome_table` 

Example Commands
================

.. code-block:: bash


	# Example command correlating county level outcomes with PA twitter data, using Facebook topics and controlling for region
	dlatkInterface.py -d paHealth -t msgsPA_2012 -c cnty -f 'feat$cat_met_a30_2000_cp_w$msgsPA_2012$cnty$16to16' --outcome_table outcome_data_with_controls --outcomes diab_perc HIV_rate PAAM_age_adj_mort IM_rate CM_rate food_insec_perc LATHF_perc MV_mort_rate DP_mort_rate UA_perc UC_perc health_care_cost NDDTC_perc OPCP_rate HI_income free_lunch_perc HR_rate --topic_tagcloud --make_topic_wordcloud --topic_lexicon met_a30_2000_freq_t50ll --p_correction simes --output_name /localdata/paHealth/a19_d6_s4 --controls 'new_england' 'midatlantic' 'south' 'midwest' 'southwest' 'west' --tagcloud_colorscheme blue


	# Example using mediation analysis
	dlatkInterface.py -d twitterGH -t messages_en -c cty_id -f 'feat$cat_met_a30_2000_cp_w$messages_en$cty_id$16to16' --mediation --path_start '1051' --outcomes 'hsgradHC03_VC93ACS3yr$10' --mediators 'hsgradHC03_VC93ACS3yr$10' --outcome_table nejm_intersect_small50k