.. _tut_output:
==============
Output Formats
==============

Summary of output types available in DLATK. Most of the following commands need the :doc:`../fwinterface/fwflag_output_name` flag which specifies the output directory.

In the csv-type outputs the DLATK namespace will be dumped as the first line. This allows you to revisit results and know your exact settings.

CSV
===

Correlations
------------

* :doc:`../fwinterface/fwflag_csv`: 
* :doc:`../fwinterface/fwflag_sort` (optional): 

.. code-block:: bash
	
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --correlate --csv --outcome_table blog_outcomes --outcomes age gender is_student is_education is_technology --outcome_with_outcome_only --output ~/correlations

.. code-block:: bash

	feature,age,p,N,CI_l,CI_u,freq,gender,p,N,CI_l,CI_u,freq,is_education,p,N,CI_l,CI_u,freq,is_student,p,N,CI_l,CI_u,freq,is_technology,p,N,CI_l,CI_u,freq
	outcome_age,1.0,0.0,1000,1.0,1.0,1000,0.013089769277,0.679288049345,1000,-0.048943029258,0.0750219732934,1000,0.172756193994,9.62582613937e-08,1000,0.111962191762,0.232261824573,1000,-0.45911352125,6.87199798082e-53,1000,-0.50668539378,-0.408754348795,1000,0.117238226305,0.000337894917181,1000,0.0556496028558,0.177938062241,1000
	outcome_gender,0.013089769277,0.679288049345,1000,-0.048943029258,0.0750219732934,1000,1.0,0.0,1000,1.0,1.0,1000,-0.161206890368,4.95925394952e-07,1000,-0.220991447954,-0.100215331046,1000,-0.0221719208099,0.483710711985,1000,-0.0840494768068,0.0398759714239,1000,0.065154589658,0.0492504223304,1000,0.00317432871342,0.126636171693,1000
	outcome_is_education,0.172756193994,6.41721742624e-08,1000,0.111962191762,0.232261824573,1000,-0.161206890368,7.43888092428e-07,1000,-0.220991447954,-0.100215331046,1000,1.0,0.0,1000,1.0,1.0,1000,-0.14424303335,7.7633413628e-06,1000,-0.204408282607,-0.0829920718038,1000,-0.0537304778134,0.0894683992084,1000,-0.115339374305,0.00829021870801,1000
	outcome_is_student,-0.45911352125,6.87199798082e-53,1000,-0.50668539378,-0.408754348795,1000,-0.0221719208099,0.604638389981,1000,-0.0840494768068,0.0398759714239,1000,-0.14424303335,5.8225060221e-06,1000,-0.204408282607,-0.0829920718038,1000,1.0,0.0,1000,1.0,1.0,1000,-0.141292966419,1.82280081643e-05,1000,-0.20152088984,-0.0800006243256,1000
	outcome_is_technology,0.117238226305,0.000253421187886,1000,0.0556496028558,0.177938062241,1000,0.065154589658,0.0656672297739,1000,0.00317432871342,0.126636171693,1000,-0.0537304778134,0.0894683992084,1000,-0.115339374305,0.00829021870801,1000,-0.141292966419,9.11400408216e-06,1000,-0.20152088984,-0.0800006243256,1000,1.0,0.0,1000,1.0,1.0,1000

The :doc:`../fwinterface/fwflag_sort` flag creates a second table where each column is sort by descending effect size and appends the table to the bottom of the csv:

.. code-block:: bash

	SORTED:
	rank,age,r,p,N,CI_l,CI_u,freq,gender,r,p,N,CI_l,CI_u,freq,is_education,r,p,N,CI_l,CI_u,freq,is_student,r,p,N,CI_l,CI_u,freq,is_technology,r,p,N,CI_l,CI_u,freq
	1,outcome_age,1.0,0.0,1000,1.0,1.0,1000,outcome_gender,1.0,0.0,1000,1.0,1.0,1000,outcome_is_education,1.0,0.0,1000,1.0,1.0,1000,outcome_is_student,1.0,0.0,1000,1.0,1.0,1000,outcome_is_technology,1.0,0.0,1000,1.0,1.0,1000
	2,outcome_is_education,0.172756193994,6.41721742624e-08,1000,0.111962191762,0.232261824573,1000,outcome_is_technology,0.065154589658,0.0656672297739,1000,0.00317432871342,0.126636171693,1000,outcome_age,0.172756193994,9.62582613937e-08,1000,0.111962191762,0.232261824573,1000,outcome_gender,-0.0221719208099,0.483710711985,1000,-0.0840494768068,0.0398759714239,1000,outcome_age,0.117238226305,0.000337894917181,1000,0.0556496028558,0.177938062241,1000
	3,outcome_is_technology,0.117238226305,0.000253421187886,1000,0.0556496028558,0.177938062241,1000,outcome_age,0.013089769277,0.679288049345,1000,-0.048943029258,0.0750219732934,1000,outcome_is_technology,-0.0537304778134,0.0894683992084,1000,-0.115339374305,0.00829021870801,1000,outcome_is_technology,-0.141292966419,9.11400408216e-06,1000,-0.20152088984,-0.0800006243256,1000,outcome_gender,0.065154589658,0.0492504223304,1000,0.00317432871342,0.126636171693,1000
	4,outcome_gender,0.013089769277,0.679288049345,1000,-0.048943029258,0.0750219732934,1000,outcome_is_student,-0.0221719208099,0.604638389981,1000,-0.0840494768068,0.0398759714239,1000,outcome_is_student,-0.14424303335,5.8225060221e-06,1000,-0.204408282607,-0.0829920718038,1000,outcome_is_education,-0.14424303335,7.7633413628e-06,1000,-0.204408282607,-0.0829920718038,1000,outcome_is_education,-0.0537304778134,0.0894683992084,1000,-0.115339374305,0.00829021870801,1000
	5,outcome_is_student,-0.45911352125,6.87199798082e-53,1000,-0.50668539378,-0.408754348795,1000,outcome_is_education,-0.161206890368,7.43888092428e-07,1000,-0.220991447954,-0.100215331046,1000,outcome_gender,-0.161206890368,4.95925394952e-07,1000,-0.220991447954,-0.100215331046,1000,outcome_age,-0.45911352125,6.87199798082e-53,1000,-0.50668539378,-0.408754348795,1000,outcome_is_student,-0.141292966419,1.82280081643e-05,1000,-0.20152088984,-0.0800006243256,1000

Predictions
-----------

Output prediction values to a csv. 

* :doc:`../fwinterface/fwflag_prediction_csv`: 

.. code-block:: bash
	
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id  -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16'  --combo_test_regression  --outcome_table blog_outcomes --outcomes age --output ~/prediction.values --prediction_csv

The above command produces the file `prediction.values.predicted_data.csv`:

.. code-block:: bash

	Id,age__withLanguage
	2863105,25.678967347
	3155970,25.0352946253
	671748,27.6127094761
	4184069,24.1537464257
	4016135,25.6544622564
	3485704,22.359848659
	3321866,18.403504235
	...

Probabilities
-------------

Output classification probabilities to a csv. 

* :doc:`../fwinterface/fwflag_probability_csv`: 

.. code-block:: bash
	
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id  -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16'  --combo_test_classif  --outcome_table blog_outcomes --outcomes is_student is_education is_technology --output ~/probability.values --probability_csv

The above command produces the file `probability.values.prediction_probabilities.csv`:

.. code-block:: bash

	Id,is_education__withLanguage,is_student__withLanguage,is_technology__withLanguage
	2863105,-0.995385541164,-0.606670069561,-1.04686722815
	3155970,-1.02199555614,-0.364230235902,-0.945148184099
	671748,-1.03019054224,-0.605114549867,-0.95579279429
	4184069,-1.04002299128,-0.673432284751,-1.26935520215
	4016135,-0.972723923762,-0.65093645124,-0.860030650986
	3485704,-0.957358049868,-0.334117399851,-1.43404698847
	...

Prediction / Classification Metrics
-----------------------------------

Output prediction or classification performance metrics to a csv delimited by `|`. 

* :doc:`../fwinterface/fwflag_csv`: 

.. code-block:: bash
	
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id  -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16'  --combo_test_regression  --outcome_table blog_outcomes --outcomes age --output ~/regression.metrics --csv

The above command produces the file `regression.metrics.variance_data.csv`:

.. code-block:: bash

	row_id|outcome|model_controls|mae|mae_folds|mse|mse_folds|N|num_features|R|r|R2|R2_folds|r_folds|r_p|r_p_folds|rho|rho_p|se_mae_folds|se_mse_folds|se_R2_folds|se_r_folds|se_r_p_folds|se_train_mean_mae_folds|test_size|train_mean_mae|train_mean_mae_folds|train_size|{model_desc}|{modelFS_desc}|w/ lang.
	1|age|()1|4.83561091756|4.83367105025|42.8168508661|42.7795793244|978|2000|0.617025365643|0.631052186337|0.380720301847|0.382388363028|0.634404850072|9.32342564224e-110|7.83089815326e-20|0.681655428515|1.40167196195e-134|0.175281125172|3.03944440616|0.0193760958564|0.0165877776996|6.98930451373e-20|0.123031255372|198|3.33229129841|6.45703007082|780|RidgeCV(alphas=array([ 1.00000e+00,  1.00000e-02,  1.00000e-04,  1.00000e+02,     1.00000e+04,  1.00000e+06]),   cv=None, fit_intercept=True, gcv_mode=None, normalize=False,   scoring=None, store_cv_values=False)|None|1
	

HTML
====

Correlation Matrix
------------------

* :doc:`../fwinterface/fwflag_rmatrix`: 
* :doc:`../fwinterface/fwflag_sort` (optional): 

.. code-block:: bash
	
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --correlate --rmatrix --outcome_table blog_outcomes --outcomes age gender is_student is_education is_technology --outcome_with_outcome_only --output ~/correlations

`Here <http://http://dlatk.wwbp.org/_static/correlations.html>`_ is HTML output from the above command. 

Word Clouds
===========

1 to 3 Gram Clouds
------------------

* :doc:`../fwinterface/fwflag_tagcloud`: Produces data for making Wordle tag clouds, saved in text files
* :doc:`../fwinterface/fwflag_make_wordclouds`: Creates pngs from the text file output

.. code-block:: bash
	
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$msgs$user_id$16to16' --tagcloud --make_wordclouds --outcome_table blog_outcomes --outcomes age gender is_student is_education is_technology --output ~/1to3grams

creates the following files:

.. code-block:: bash

	1to3grams_tagcloud_wordclouds/age_pos.png
	1to3grams_tagcloud_wordclouds/age_neg.png
	1to3grams_tagcloud_wordclouds/gender_1_pos.png
	1to3grams_tagcloud_wordclouds/gender_1_neg.png
	1to3grams_tagcloud_wordclouds/is_student_pos.png
	1to3grams_tagcloud_wordclouds/is_student_neg.png
	1to3grams_tagcloud_wordclouds/is_education_pos.png
	1to3grams_tagcloud_wordclouds/is_education_neg.png
	1to3grams_tagcloud_wordclouds/is_technology_pos.png

Topic Clouds
------------

* :doc:`../fwinterface/fwflag_topic_tagcloud`: Produces data for making topic Wordles, saved in text files
* :doc:`../fwinterface/fwflag_make_topic_wordclouds`: Creates pngs from the text file output 

.. code-block:: bash
	
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16'  --topic_tagcloud --make_topic_wordclouds  --outcome_table blog_outcomes --outcomes age gender is_student is_education is_technology --output ~/fbtopics --topic_lexicon met_a30_2000_freq_t50ll

creates the following directories where the topic clouds are printed:

.. code-block:: bash
	
	fbtopics_topic_tagcloud_wordclouds/age
	fbtopics_topic_tagcloud_wordclouds/gender 
	fbtopics_topic_tagcloud_wordclouds/is_student
	fbtopics_topic_tagcloud_wordclouds/is_education
	fbtopics_topic_tagcloud_wordclouds/is_technology

Print Word Clouds For A Topic Lexicon 
-------------------------------------

Create a word clound png for every topic in a given topic lexicon.

* :doc:`../fwinterface/fwflag_make_all_topic_wordclouds`

.. code-block:: bash
	
	dlatkInterface.py --topic_lexicon met_a30_2000_freq_t50ll --output all_met_a30_2000_tagclouds --make_all_topic_wordclouds

Ini files
=========

See the :doc:`../tutorials/tut_ini_files` tutorial for more info.

Pickles
=======

Save a predictive model to a pickle file with:

* :doc:`../fwinterface/fwflag_picklefile`
* :doc:`../fwinterface/fwflag_save_model`

.. code-block:: bash
	
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$msgs$user_id$16to16' --outcome_table blog_outcomes --outcomes age --train_regression --save_model --picklefile age.pickle

Load a predictive model to a pickle file with:

* :doc:`../fwinterface/fwflag_picklefile`
* :doc:`../fwinterface/fwflag_load_model`

.. code-block:: bash
	
	# Loads the regression model in age.pickle, and uses the features to predict the ages of the users in 
	# blog_outcomes, and compares the predicted ages to the actual ages in the table.

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$msgs$user_id$16to16' --outcome_table blog_outcomes --outcomes age --train_regression --load_model --picklefile age.pickle

