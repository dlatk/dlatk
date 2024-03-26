.. _fwflag_make_topic_wordclouds:
=======================
--make_topic_wordclouds
=======================
Switch
======

--make_topic_wordclouds

Description
===========

Makes word clouds from topcs. This needs an output topic tagcloud file. For NON topic word clouds see --make_wordclouds.

Argument and Default Value
==========================

None

Other Switches
==============

Required Switches:

* :doc:`fwflag_topic_tagcloud` or :doc:`fwflag_corp_topic_tagcloud` 

Optional Switches:

* :doc:`fwflag_tagcloud_colorscheme` 

Example Commands
================

.. code-block:: bash

	# Example command correlating county level outcomes with PA twitter data, using Facebook topics and controlling for region
	 ./dlatkInterface.py -d paHealth -t msgsPA_2012 -c cnty -f 'feat$cat_met_a30_2000_cp_w$msgsPA_2012$cnty$16to16' outcome_table outcome_data_with_controls --outcomes diab_perc HIV_rate PAAM_age_adj_mort IM_rate CM_rate food_insec_perc LATHF_perc MV_mort_rate DP_mort_rate --topic_tagcloud --make_topic_wordcloud --topic_lexicon met_a30_2000_freq_t50ll --p_correction simes --output_name /localdata/paHealth/a19_d6_s4 --controls 'new_england' 'midatlantic' 'south' 'midwest' 'southwest' 'west' --tagcloud_colorscheme blue --group_freq_thresh 40000