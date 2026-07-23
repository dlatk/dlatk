.. _fwflag_cca_permute:
=============
--cca_permute
=============
Switch
======

--cca_permute

Description
===========

Finds the best penalties for performing CCA.

Argument and Default Value
==========================

Number  of iterations to do permutations on.

Details
=======

This switch does a number of iteration to find the penalties best fitting for CCA on the current features/outcomes/controls (ccaPermute from the R PMA package).
As in CCA, This first removes groups that have at least 4 non null values in the feature or outcome matrix, then performs softImpute (matrix completion) to get rid of the null values, and then iterates.

It will print for each penalty the number of non-zero outcomes/features per component, so this can help choose a sparsity constraint.

If you want to do cca_permute without using the features, use :doc:`fwflag_cca_outcomes_vs_controls` as an additional flag.


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`
* :doc:`fwflag_f`
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` 

Optional Switches:

* :doc:`fwflag_group_freq_thresh`
* :doc:`fwflag_outcome_controls`
* :doc:`fwflag_cca_outcomes_vs_controls` 

Example Commands
================

.. code-block:: bash


	# Finds best CCA penalties for the topics and 15 diseases.
	dlatkInterface.py -d county_disease -t messages_en -c cnty -f 'feat$cat_met_a30_2000_cp_w$messages_en$cnty$16to16' --group_freq_thresh 50000  --outcome_table topDeaths_comp_0910 --outcomes 01hea_aar 02mal_aar 03chr_aar 04cer_aar 05acc_aar 06alz_aar 07dia_aar 08nep_aar 09flu_aar 10sel_aar 11sep_aar 12liv_aar 13hyp_aar 14par_aar 15pne_aar --cca_permute 25
