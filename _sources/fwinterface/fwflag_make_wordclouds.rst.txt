.. _fwflag_make_wordclouds:
=================
--make_wordclouds
=================

Switch
======

--make_wordclouds

Description
===========

Makes word clouds. This needs an output topic tagcloud file. For topic word clouds see --make_topic_wordclouds.

Argument and Default Value
==========================

None

Other Switches
==============

Required Switches:
* :doc:`fwflag_topic_tagcloud` Optional Switches:
* :doc:`fwflag_tagcloud_colorscheme` 

Example Commands
================
.. code:doc:`fwflag_block`:: python

	# Example command correlating county level outcomes with PA twitter data, using Facebook topics and controlling for region
	 ./fwInterface.py -d paHealth -t msgsPA_2012 -c cnty -f 'feat$1gram$msgsPA_2012$cnty$16to16' --outcome_table outcome_data_with_controls --outcomes diab_perc HIV_rate PAAM_age_adj_mort IM_rate CM_rate food_insec_perc LATHF_perc MV_mort_rate DP_mort_rate --tagcloud --make_wordclouds 
	 --output_name /localdata/paHealth/d6 --controls 'new_england' 'midatlantic' 'south' 'midwest' 'southwest' 'west'