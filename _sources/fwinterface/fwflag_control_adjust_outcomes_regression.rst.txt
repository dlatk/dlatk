.. _fwflag_control_adjust_outcomes_regression:
====================================
--control_adjust_outcomes_regression
====================================
Switch
======

--control_adjust_outcomes_regression, --control_adjust_reg

Description
===========

Use controls to predict outcomes and produces adjusted outcomes

Argument and Default Value
==========================

None

Details
=======

This switch will run in K-folds (:doc:`fwflag_folds`), and for each fold will train a prediction model (on K-1/K % of the groups) using the controls as features, and predict on the 1/K % of groups.

Use :doc:`fwflag_pred_csv` to write the predicted data to a csv (will end in .predicted_data.csv). To use the data as outcomes after that, usually we import the csv into MySQL.

Use :doc:`fwflag_csv` to write the variance data to csv (i.e. how well we do at predicting the outcomes from the controls.)

See :doc:`fwflag_train_regression` and :doc:`fwflag_combo_test_regression` for info on how folds work and for model selection and feature selection.


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` 

Optional Switches:

* :doc:`fwflag_group_freq_thresh`
* :doc:`fwflag_model`
* :doc:`fwflag_folds`
* :doc:`fwflag_pred_csv`
* :doc:`fwflag_no_standardize`
* :doc:`fwflag_sparse`
* :doc:`fwflag_regression_to_lexicon` etc.

Example Commands
================

.. code-block:: bash


	# Uses all 10 controls to predict the top15 causes of deaths, and saves the predictions from the model into the csv file
	# named top15aar.d6v4adj.predicted_data.csv. It also saved the model performances into top15aar.d6v4adj.variance_data.csv
	# Only uses all controls because :doc:`fwflag_combo_sizes` is the size of the list of controls.
	dlatkInterface.py -d county_disease -t msgs_2011to13 -c cnty --group_freq_thresh 50000 --outcome_table topDeaths_comp_11to13 --outcomes 01hea_aar 02mal_aar 03res_aar 04acc_aar 05cer_aar 06alz_aar 07dia_aar 08flu_aar 09nep_aar 10sui_aar 11sep_aar 12liv_aar 13hyp_aar 14par_aar 15pne_aar --outcome_controls 'hsgradHC03_VC93ACS3yr$10' 'bachdegHC03_VC94ACS3yr$10' 'logincomeHC01_VC85ACS3yr$10' 'unemployAve_BLSLAUS$0910' 'femalePOP165210D$10' 'hispanicPOP405210D$10' 'blackPOP255210D$10' 'forgnbornHC03_VC134ACS3yr$10' county_density 'marriedaveHC03_AC3yr$10' --control_adjust_reg --folds 10 --csv --pred_csv --output_name /localdata/county_disease/SQL_csvs/top15aar.d6v4adj --combo_sizes 10 
