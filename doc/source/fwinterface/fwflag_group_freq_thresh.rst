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

Argument is an integer number. Unless specified the default value is set based on the :doc:`fwflag_c` flag: message level gft = 1, user level gft = 1000, county level gft = 40000.

Details
=======

Counts the number of words in each group specified by correl_field. If this count is less than the given group frequency threshold then this group is thrown out. The group is otherwise kept. 


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 

Example Commands
================

.. code-block:: bash


	# Example command correlating county level outcomes with PA twitter data, using Facebook topics and controlling for region
	dlatkInterface.py -d paHealth -t msgsPA_2012 -c cnty -f 'feat$cat_met_a30_2000_cp_w$msgsPA_2012$cnty$16to16' --outcome_table outcome_data_with_controls --outcomes diab_perc HIV_rate PAAM_age_adj_mort IM_rate CM_rate food_insec_perc LATHF_perc MV_mort_rate DP_mort_rate UA_perc UC_perc health_care_cost NDDTC_perc OPCP_rate HI_income free_lunch_perc HR_rate --topic_tagcloud --make_topic_wordcloud --topic_lexicon met_a30_2000_freq_t50ll --p_correction simes --output_name /localdata/paHealth/a19_d6_s4 --controls` 'new_england' 'midatlantic' 'south' 'midwest' 'southwest' 'west' --tagcloud_colorscheme blue --group_freq_thresh 40000


	# Create a set of 1-grams from the Primals data at the message level
	dlatkInterface.py -d primals -t primals_new -c message_id --add_ngrams -n 1 --feat_occ_filter --set_p_occ 0.001 --group_freq_thresh 1000