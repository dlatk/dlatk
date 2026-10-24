.. _fwflag_cca:
=====
--cca
=====
Switch
======

--cca

Description
===========

Performs CCA on the features and the outcomes and generates clusters (called components).

Argument and Default Value
==========================

Number of components to generate.

Details
=======

This switch performs finds components that explain variance in both the features and the outcomes&controls, using the sparse CCA implementation in R. This first removes groups that have at least 4 non null values in the feature or outcome matrix, then performs softImpute (matrix completion) to get rid of the null values, and then performs CCA. Output will be in the form of (features x component) weights and (outcomes x component) weights, and the exact output format depends on the flags you specify (:doc:`fwflag_rmatrix` or :doc:`fwflag_csv` etc.).

Note that the number of components (the argument, sometimes called K) must satisfy:


There are a bunch of parameters for the sparse CCA function in R, the only ones that have a command line switch are penalties set on the "left" matrix (aka X, usually features) and the "right" matrix (aka Z, usually outcomes):
:doc:`fwflag_cca_penalty_feats`, aka :doc:`fwflag_cca_penaltyx` :doc:`fwflag_cca_penalty_outcomes`, aka :doc:`fwflag_cca_penaltyz` To find which values to assign to these parameters, you can run :doc:`fwflag_cca_permute` 
If you want to do cca without using the features, use :doc:`fwflag_cca_outcomes_vs_controls` as an additional flag.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`
* :doc:`fwflag_f`
* :doc:`fwflag_outcome_table`
* :doc:`fwflag_outcomes` 

Optional Switches:

* :doc:`fwflag_group_freq_thresh`
* :doc:`fwflag_outcome_controls`
* :doc:`fwflag_cca_penalty_feats`
* :doc:`fwflag_cca_penalty_outcomes`
* :doc:`fwflag_cca_outcomes_vs_controls`
* :doc:`fwflag_topic_tagcloud` etc.

Example Commands
================

.. code-block:: bash


	 # Performs CCA on the topics and 15 diseases using 15 components
	 # Will output 2 HTML files called d0s0.K15.outcomes.html and d0s0.K15.feat.html
	 dlatkInterface.py -d county_disease -t messages_en -c cnty -f 'feat$cat_met_a30_2000_cp_w$messages_en$cnty$16to16' --group_freq_thresh 50000 --outcome_table topDeaths_comp_0910 --outcomes 01hea_aar 02mal_aar 03chr_aar 04cer_aar 05acc_aar 06alz_aar 07dia_aar 08nep_aar 09flu_aar 10sel_aar 11sep_aar 12liv_aar 13hyp_aar 14par_aar 15pne_aar --output_name ~/CCA/d0s0.K15 --rmatrix --cca 15 

	 # Performs CCA on the topics and 15 diseases using 15 components, with penalties set to 0.5 and 0.5 for X and Z.
	 # Will output 2 HTML files called d0s0.K15.X0_5.Z0_5.outcomes.html and d0s0.K15.X0_5.Z0_5.feat.html, plus d0s0.K15.X0_5.Z0_5_topic_tagcloud.txt.
	 # It will also create the topic wordclouds in a separate directory. 
	 dlatkInterface.py -d county_disease -t messages_en -c cnty -f 'feat$cat_met_a30_2000_cp_w$messages_en$cnty$16to16' --group_freq_thresh 50000 --outcome_table topDeaths_comp_0910 --outcomes 01hea_aar 02mal_aar 03chr_aar 04cer_aar 05acc_aar 06alz_aar 07dia_aar 08nep_aar 09flu_aar 10sel_aar 11sep_aar 12liv_aar 13hyp_aar 14par_aar 15pne_aar --output_name ~/CCA/d0s0.K15.X0_5.Z0_5 --rmatrix --cca 15 --topic_tagcloud --topic_lexicon met_a30_2000_freq_t50ll --cca_penaltyx .5 --cca_penaltyz .5 --csv --sort --make_topic_wordclouds 