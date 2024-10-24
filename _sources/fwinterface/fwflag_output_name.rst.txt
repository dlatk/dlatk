.. _fwflag_output_name:
=============
--output_name
=============
Switch
======

--output_name FILENAME

Description
===========

Overrides the default filename for output.

Argument and Default Value
==========================

String for output file name.

Details
=======

Aliases: --output, --output_file

Other Switches
==============

This is optional for many switches. 

* :doc:`../fwinterface/fwflag_rmatrix` 
* :doc:`../fwinterface/fwflag_csv` 
* :doc:`../fwinterface/fwflag_prediction_csv` 
* :doc:`../fwinterface/fwflag_probability_csv`
* :doc:`../fwinterface/fwflag_tagcloud` and :doc:`../fwinterface/fwflag_make_wordclouds`
* :doc:`../fwinterface/fwflag_topic_tagcloud` and :doc:`../fwinterface/fwflag_make_topic_wordclouds`

Example Commands
================

This commands will create topic word clouds correlated with age and gender and place them in the `age_and_gender_clouds` directory:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id \ 
	-f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' \ 
	 --outcome_table blog_outcomes  --group_freq_thresh 500 \ 
	 --outcomes age gender --output_name age_and_gender_clouds \ 
	 --topic_tagcloud --make_topic_wordcloud --topic_lexicon met_a30_2000_freq_t50ll \ 
	--tagcloud_colorscheme bluered

