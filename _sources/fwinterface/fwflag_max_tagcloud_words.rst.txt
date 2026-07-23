.. _fwflag_max_tagcloud_words:
====================
--max_tagcloud_words
====================
Switch
======

--max_tagcloud_words

Description
===========

Specify the maximum number of words to appear in a tagcloud.

Argument and Default Value
==========================

Integer value. Default value is 100.

Other Switches
==============

Required Switches:

* :doc:`fwflag_tagcloud` 

Example Commands
================
.. code:doc:`fwflag_block`:: python

	# Example command correlating county level outcomes with PA twitter data, using Facebook topics and controlling for region
	 ./dlatkInterface.py -d paHealth -t msgsPA_2012 -c cnty -f 'feat$1gram$msgsPA_2012$cnty$16to16' --outcome_table outcome_data_with_controls --outcomes diab_perc HIV_rate PAAM_age_adj_mort IM_rate CM_rate food_insec_perc LATHF_perc MV_mort_rate DP_mort_rate --tagcloud --make_wordclouds --output_name /localdata/paHealth/d6 --controls 'new_england' 'midatlantic' 'south' 'midwest' 'southwest' 'west' --max_tagcloud_words 25