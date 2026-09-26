.. _fwflag_tagcloud:
==========
--tagcloud
==========
Switch
======

--tagcloud

Description
===========

Produces data for making Wordle tag clouds. For a topic based tagcloud see --topic_tagcloud.

Argument and Default Value
==========================

None

Details
=======Contents [hide]Switch
Description
    Argument and Default Value
    
Other Switches
==============
    
Example Commands
================
.. code:doc:`fwflag_block`:: python
Switch

:doc:`fwflag_tagcloud` 
Description

Produces data for making Wordle tag clouds. For a topic based tagcloud see :doc:`fwflag_topic_tagcloud`. 
Argument and Default Value

None


Other Switches
==============

Required Switches:
:doc:`fwflag_f`, :doc:`fwflag_feat_table` FEAT_TABLE (cannot be topic based)
:doc:`fwflag_outcome_table` :doc:`fwflag_outcomes` Optional Switches
:doc:`fwflag_make_wordclouds` :doc:`fwflag_tagcloud_filter` / :doc:`fwflag_no_tagcloud_filter` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Example command correlating county level outcomes with PA twitter data, using Facebook topics and controlling for region
 ./fwInterface.py :doc:`fwflag_d` paHealth :doc:`fwflag_t` msgsPA_2012 :doc:`fwflag_c` cnty \ 
 :doc:`fwflag_f` 'feat$1gram$msgsPA_2012$cnty$16to16' :doc:`fwflag_outcome_table` outcome_data_with_controls \ 
 :doc:`fwflag_outcomes` diab_perc HIV_rate PAAM_age_adj_mort IM_rate CM_rate food_insec_perc LATHF_perc MV_mort_rate DP_mort_rate \ 
 :doc:`fwflag_tagcloud` :doc:`fwflag_make_wordclouds`  \ 
 :doc:`fwflag_output_name` /localdata/paHealth/d6 :doc:`fwflag_controls` 'new_england' 'midatlantic' 'south' 'midwest' 'southwest' 'west'