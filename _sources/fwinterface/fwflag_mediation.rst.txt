.. _fwflag_mediation:
===========
--mediation
===========
Switch
======

--mediation

Description
===========

Mediation analysis considers three variables, as seen in the following figure: path start (also called the treatment), mediator and outcome.

Argument and Default Value
==========================

These variables can be located within a feature or outcome table. There are 4 switches to specify where the features will go within the mediation:  --feat_as_path_start, --feat_as_outcome, --feat_as_control, --no_features. Once you specify which variable is in the feature table all other variables will be in the outcome table. There is the option to specify specific features or consider the entire feature table. For example, if your outcome variable is a feature you can run either of the following two commands:

Details
=======

Mediation analysis considers three variables, as seen in the following figure: path start (also called the treatment), mediator and outcome. 

These variables can be located within a feature or outcome table. There are 4 switches to specify where the features will go within the mediation:  :doc:`fwflag_feat_as_path_start`, :doc:`fwflag_feat_as_outcome`, :doc:`fwflag_feat_as_control`, :doc:`fwflag_no_features`. Once you specify which variable is in the feature table all other variables will be in the outcome table. There is the option to specify specific features or consider the entire feature table. For example, if your outcome variable is a feature you can run either of the following two commands:

 # Specify feature 1200 as the outcome
 ./dlatkInterface.py -d twitterGH -t messages_en -c cty_id -f 'feat$cat_met_a30_2000_cp_w$messages_en$cty_id$16to16' --mediation --path_start 'hsgradHC03_VC93ACS3yr$10' --mediators 'ucd_I25_1_atheroHD$0910_ageadj' --outcome_table nejm_intersect_small50k --outcomes '1200' --feat_as_outcome

 # Consider all features as outcomes
 ./dlatkInterface.py -d twitterGH -t messages_en -c cty_id -f 'feat$cat_met_a30_2000_cp_w$messages_en$cty_id$16to16' --mediation --path_start 'hsgradHC03_VC93ACS3yr$10' --mediators 'ucd_I25_1_atheroHD$0910_ageadj' --outcome_table nejm_intersect_small50k --feat_as_outcome


 The analysis is done using two methods. The first is the Baron and Kenny method with a Sobel test for significance. We first do three regressions (all of which are normalized):

Regress the outcome with the path start: 
 
Regress the mediator with the path start:
 
Regress the outcome with the path start and the mediator: 
 
We then check if ,  and  are significant and that  is smaller in absolute value than . The Sobel test score is calculated as 
 
where  and  are the standard error of , , respectively.

The second analysis outputs the following varirables:
 ACME (control) = Average Causal Mediation Effect with controls
 ACME (treated) = Average Causal Mediation Effect without controls
 ADE (control)  = Average Direct Effect with controls
 ADE (treated)  = Average Direct Effect without controls

When no controls are given then ACME (control) = ACME (treated) and ADE (control) = ADE (treated). In the case of linear models with no interactions involving the mediator, the results should be similar or identical to the earlier Barron:doc:`fwflag_Kenny` approach.


Other Switches
==============

Required Switches:
* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 
* :doc:`fwflag_f` FEATURE_TABLE_NAME (note: this becomes optional if :doc:`fwflag_no_features` is used)
* :doc:`fwflag_path_starts` PATH_START_1 ... PATH_START_I
* :doc:`fwflag_mediators` MEDIATOR_1 ... MEDIATOR_J 
* :doc:`fwflag_outcomes` OUTCOME_1 ... OUTCOME_K
* :doc:`fwflag_outcome_table` OUTCOME_TABLE_NAME
Optional Switches:
* :doc:`fwflag_controls` CONTROL_1 ... CONTROL_L
* :doc:`fwflag_group_freq_thresh` GROUP_THRESH
* :doc:`fwflag_feat_as_path_start` or :doc:`fwflag_feat_as_outcome` or :doc:`fwflag_feat_as_control`  or :doc:`fwflag_no_features` 
* :doc:`fwflag_p_value` :doc:`fwflag_mediation_boot` :doc:`fwflag_mediation_boot_num` :doc:`fwflag_output_name` OUTPUT_FILE_NAME
* :doc:`fwflag_mediation_no_summary` 
* :doc:`fwflag_mediation_csv` 
* :doc:`fwflag_mediation_method` 
* :doc:`fwflag_no_bonferroni` 
* :doc:`fwflag_p_correction` 


Example Commands
================
.. code:doc:`fwflag_block`:: python

	 # Example
	 # Since no mediators are given, this will consider every feature in the feature table as a mediator
	 ./dlatkInterface.py -d twitterGH -t messages_en -c cty_id -f 'feat$cat_met_a30_2000_cp_w$messages_en$cty_id$16to16' --mediation --path_start 'hsgradHC03_VC93ACS3yr$10' --outcomes 'ucd_I25_1_atheroHD$0910_ageadj' --outcome_table nejm_intersect_small50k --mediation_csv --group_freq_thresh 40000 --output_name mediation_40k.csv --mediation_boot


Part of Output:

.. code:doc:`fwflag_block`:: python

	 Mediation number 2000 out of 2000
	 Path Start: hsgradHC03_VC93ACS3yr$10, Mediator: 1051, Outcome: ucd_I25_1_atheroHD$0910_ageadj
	 C: -0.134772542078, C_p: 6.8251850716e-07, C': -0.163575622445, C'_p: 2.05569898945e-08
	 C-C': 0.0288030803674, alpha*beta: 0.0288030803674
	 alpha: -0.36936035776, alpha_error: 0.0253295211266, alpha_p: 7.8905979142e-45
	 beta: -0.0779809737625, beta_error: 0.0289964186593, beta_p: 0.00724804697931
	 Sobel z-score: 2.64473000024, Sobel SE: 0.0108907451289, Sobel p: 0.00817561235516
	                            Estimate P-value Lower CI bound Upper CI bound
	 Prop. mediated (average)  -0.2121596   0.004     -0.4912299    -0.06154572
	 ACME (average)            0.02911135   0.004    0.008990151      0.0541393
	 ADE (average)             -0.1630937       0     -0.2160734     -0.1054184
	 ACME (treated)            0.02911135   0.004    0.008990151      0.0541393
	 ACME (control)            0.02911135   0.004    0.008990151      0.0541393
	 ADE (treated)             -0.1630937       0     -0.2160734     -0.1054184
	 ADE (control)             -0.1630937       0     -0.2160734     -0.1054184
	 Total effect              -0.1339823       0     -0.1848832     -0.0804331
	 Prop. mediated (treated)  -0.2121596   0.004     -0.4912299    -0.06154572
	 Prop. mediated (control)  -0.2121596   0.004     -0.4912299    -0.06154572