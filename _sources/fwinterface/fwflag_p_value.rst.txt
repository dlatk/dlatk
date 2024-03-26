.. _fwflag_p_value:
=========
--p_value
=========
Switch
======

--p_value

Description
===========

P value to use when creating tagclouds and topic tagclouds and for printing number of significant features to terminal when running correlate.

Argument and Default Value
==========================

Also used with mediation analysis: specify the significance level of results to be printed in the summary report. Any mediation results with Sobel p less than the specified value will be printed.

Details
=======

When using :doc:`fwflag_correlate` this switch will print the number of features significant at this threshold:
 #Example output
 1259 features significant at p < 1e:doc:`fwflag_05` 
Other Switches
==============

Optional Switches:
:doc:`fwflag_correlate` :doc:`fwflag_tagcloud` :doc:`fwflag_topic_tagcloud` :doc:`fwflag_DDLATagcloud`? 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Create topic tagclouds
 ./fwInterface.py :doc:`fwflag_d` county_addiction :doc:`fwflag_t` msgs_2011to13 :doc:`fwflag_c` cnty :doc:`fwflag_word_table` 'feat$1gram$msgs_2011to13$cnty$16to16$0_1' :doc:`fwflag_group_freq_thresh` 40000 :doc:`fwflag_f` \ 
 'feat$cat_met_a30_2000_cp_w$msgs_2011to13$cnty$16to16' :doc:`fwflag_outcome_table` main_interest_vars_controls :doc:`fwflag_outcomes` ExcessDrink_Percent :doc:`fwflag_no_bonf` \ 
 :doc:`fwflag_p_correction` simes :doc:`fwflag_topic_tagcloud` :doc:`fwflag_make_topic_wordclouds` :doc:`fwflag_topic_lexicon` met_a30_2000_freq_t50ll :doc:`fwflag_csv` :doc:`fwflag_output_name` ~/xxx_output :doc:`fwflag_p_value` 0.001


 # Correlate features with outcome
 ./fwInterface.py :doc:`fwflag_d` county_addiction :doc:`fwflag_t` msgs_2011to13 :doc:`fwflag_c` cnty :doc:`fwflag_word_table` 'feat$1gram$msgs_2011to13$cnty$16to16$0_1' :doc:`fwflag_group_freq_thresh` 40000 :doc:`fwflag_f` \ 
 'feat$cat_met_a30_2000_cp_w$msgs_2011to13$cnty$16to16' :doc:`fwflag_outcome_table` main_interest_vars_controls :doc:`fwflag_outcomes` ExcessDrink_Percent :doc:`fwflag_no_bonf` \ 
 :doc:`fwflag_p_correction` simes :doc:`fwflag_correlate` :doc:`fwflag_p_value` 0.001