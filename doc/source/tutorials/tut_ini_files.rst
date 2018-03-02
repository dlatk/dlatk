.. _tut_init_files:
===============
Using INI Files
===============

dlatkInterface can read parameters from an input file (or INI file) using the flag :doc:`../fwinterface/fwflag_from_file`. These are useful in shortening your commands and keeping track of parameters you've used in past projects. The **fromFile()** method in *FeatureGetter*, *outcomeGetter* and *FeatureStar* can also read these files. 

In general, setup level flags are stored here. For example: database name, message table name, p correction method. Types of analysis (correlation, prediction, wordcloud creation) are not stored in these files. 

Creating an init file from the command line
-------------------------------------------
See the :doc:`../fwinterface/fwflag_to_file` flag. Here is an example:

The command:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 \ 
	-f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' --outcome_table blog_outcomes \ 
	--outcomes is_student --controls age gender --correlate  \ 
	--output_name ~/my_output --p_value 0.05 --p_correction 'bonferroni' --to_file ~/someFile.ini

Creates the file *someFile.ini*:

.. code-block:: bash

	[constants]
	correl_field = user_id
	feattable = feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16
	outcometable = blog_outcomes
	outcomefields = is_student
	outcomecontrols = age, gender
	outputname = /home/username/my_output
	groupfreqthresh = 500
	p_correction_method = bonferroni

Using an init file via the command line
---------------------------------------

The above command then shortens to:

.. code-block:: bash

	dlatkInterface.py --from_file ~/someFile.ini --correlate

Using an init file with classes
-------------------------------

The above file then shortens your dlatkInterface calls:

.. code-block:: python

	from dlatk.featureStar import FeatureStar
	fs = FeatureStar.fromFile('~/someFile.ini')

For more examples of using ini files with dlatk classes: :doc:`tut_classes`