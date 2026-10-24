.. _fwflag_cca_outcomes_vs_controls:
==========================
--cca_outcomes_vs_controls
==========================
Switch
======

--cca_outcomes_vs_controls

Description
===========

Performs CCA on the outcomes and the controls, not using the features.

Argument and Default Value
==========================

None

Details
=======

This switch causes the infrastructure to perform CCA on the two following matrices:
Groups by outcomes
Groups by controls
Groups that don't have values for at least 4 columns will be deleted (you can change the 4 into whatever you want, but it's in clustering.py or in dlatkInterface.py).
Unless changed in the code, the two matrices are concatenated and softImpute is done on the concatenated matrix, and then split back into outcomes and controls.

Note that X is now the controls matrix, so use --cca_penatlyx to change the penalty on that matrix.

Change the two following lines (clustering.py) if you want softImpute to be done on the matrices separately:

.. code-block:: bash

	# X, Z, Xfreqs, Zfreqs = self.prepMatrices(Xdf,Zdf, NAthresh = NAthresh, softImputeXtoo=True)                                                                                                        
	X, Z, Xfreqs, Zfreqs = self.prepMatricesTogether(Xdf,Zdf, NAthresh = NAthresh)

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`
* :doc:`fwflag_outcomes`, :doc:`fwflag_outcome_controls`
* :doc:`fwflag_cca` 

Optional Switches:

* :doc:`fwflag_group_freq_thresh`
* :doc:`fwflag_cca_penalty_feats`
* :doc:`fwflag_cca_penalty_outcomes`
* :doc:`fwflag_save_model`
* :doc:`fwflag_picklefile` 

Example Commands
================

.. code-block:: bash


	# This performs CCA on the top 15 causes of death VS the 10 SES and demographic controls, with .4 penalties. 
	# This also saves the component distributions to the picklefile, for further use.
	dlatkInterface.py -d county_disease -t messages_en -c cnty -f feat$cat_met_a30_2000_cp_w$messages_en$cnty$16to16 --group_freq_thresh 0 --outcome_table topDeaths_comp_0910 --outcomes 01hea_aar 02mal_aar 03chr_aar 04cer_aar 05acc_aar 06alz_aar 07dia_aar 08nep_aar 09flu_aar 10sel_aar 11sep_aar 12liv_aar 13hyp_aar 14par_aar 15pne_aar --outcome_controls hsgradHC03_VC93ACS3yr$10 bachdegHC03_VC94ACS3yr$10 logincomeHC01_VC85ACS3yr$10 unemployAve_BLSLAUS$0910 femalePOP165210D$10 hispanicPOP405210D$10 blackPOP255210D$10 forgnbornHC03_VC134ACS3yr$10 county_density marriedaveHC03_AC3yr$10 --cca 10 --output_name diseasesOnd6s4.K10.X0_4.Z0_4.gft0 --rmatrix --cca_penaltyx .4 --cca_penaltyz .4 --csv --sort --cca_outcomes_vs_controls --save_model --picklefile diseasesOnd6s4.K10.X0_4.Z0_4.gft0.pickle
