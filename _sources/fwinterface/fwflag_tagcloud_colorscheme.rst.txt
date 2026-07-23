.. _fwflag_tagcloud_colorscheme:
======================
--tagcloud_colorscheme
======================
Switch
======

--tagcloud_colorscheme

Description
===========

Specify a color scheme to use for tagcloud generation.

Argument and Default Value
==========================

Name of color: red, blue, redblue, bluered, and red-random. Default is multi. bluered prints positively correlated clouds as blue and negatively correlated clouds as red, similarly, redblue prints positively correlated clouds as red and negatively correlated clouds as blue.
    
Other Switches
==============

Required Switches (one of the following):

* :doc:`fwflag_tagcloud`
* :doc:`fwflag_topic_tagcloud`
* :doc:`fwflag_DDLATagcloud`
    
Example Commands
================
.. code-block:: bash


 # Example command correlating county level outcomes with PA twitter data, using Facebook topics and controlling for region
 ./dlatkInterface.py :doc:`fwflag_d` paHealth :doc:`fwflag_t` msgsPA_2012 :doc:`fwflag_c` cnty \
 :doc:`fwflag_f` 'feat$1gram$msgsPA_2012$cnty$16to16' :doc:`fwflag_outcome_table` outcome_data_with_controls \
 :doc:`fwflag_outcomes` diab_perc HIV_rate PAAM_age_adj_mort IM_rate CM_rate food_insec_perc LATHF_perc MV_mort_rate DP_mort_rate \
 :doc:`fwflag_tagcloud` :doc:`fwflag_make_wordclouds`  \
 :doc:`fwflag_output_name` /localdata/paHealth/d6 :doc:`fwflag_controls` 'new_england' 'midatlantic' 'south' 'midwest' 'southwest' 'west' \
 :doc:`fwflag_tagcloud_colorscheme` blue
