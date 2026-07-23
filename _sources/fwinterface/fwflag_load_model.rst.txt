.. _fwflag_load_model:
============
--load_model
============
Switch
======

--load_model

Description
===========

Loads the model that is stored in a file.

Argument and Default Value
==========================

None

Details
=======

If you are running a command that uses a model that was trained previously, using this will tell the infrastructure to load that model. This switch requires the use of :doc:`fwflag_picklefile`. In order to use this, you must have saved the model using :doc:`fwflag_save_model`.
See one of these three for more info:
:doc:`fwflag_train_regression` :doc:`fwflag_train_classifiers` :doc:`fwflag_cca`

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`,
* :doc:`fwflag_f`,
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes`
* :doc:`fwflag_picklefile`

Optional Switches:
* :doc:`fwflag_predict_regression` and anything optional to that
* :doc:`fwflag_predict_classifiers` and anything optional to that
* :doc:`fwflag_cca` and anything optional to that
etc.

Example Commands
================
.. code-block:: doc:`fwflag_block`:: python


 # Loads the regression model in deleteMe.pickle, and uses the features to predict the ages of the users in
 # masterstats_andy_r10k, and compares the predicted ages to the actual ages in the table.
 ~/fwInterface.py :doc:`fwflag_d` fb20 :doc:`fwflag_t` messages_en :doc:`fwflag_c` user_id :doc:`fwflag_f` 'feat$1gram$messages_en$user_id$16to16$0_01'
 :doc:`fwflag_outcome_table` masterstats_andy_r10k :doc:`fwflag_outcomes` age :doc:`fwflag_load_model` :doc:`fwflag_picklefile` deleteMe.pickle
 :doc:`fwflag_predict_regression` 
 # Loads CCA model and predicts component distribution on counties using the diseases view.
 # Inserts that distribution into the DELETEME SQL table that it will create.
 ~/fwInterface.py :doc:`fwflag_d` county_disease :doc:`fwflag_t` messages_en :doc:`fwflag_c` cnty :doc:`fwflag_f` 'feat$cat_met_a30_2000_cp_w$messages_en$cnty$16to16'
 :doc:`fwflag_outcome_table` topDeaths_comp_0910 :doc:`fwflag_outcomes` 01hea_aar 02mal_aar 03chr_aar 04cer_aar 05acc_aar 06alz_aar 07dia_aar
 08nep_aar 09flu_aar 10sel_aar 11sep_aar 12liv_aar 13hyp_aar 14par_aar 15pne_aar  :doc:`fwflag_cca_predict_components`
 :doc:`fwflag_load_model` :doc:`fwflag_picklefile` diseasesOnd6s4.K10.X0_4.Z0_4.gft0.pickle :doc:`fwflag_to_sql_table` DELETEME
