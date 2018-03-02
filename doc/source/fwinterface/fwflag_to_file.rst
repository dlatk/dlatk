.. _fwflag_to_file:
=========
--to_file
=========
Switch
======

--to_file [FILENAME]

Description
===========

Creates a text file to be used when calling the fromFile() method in FeatureGetter or OutcomeGetter. See also: :doc:`fwflag_from_file`. See :doc:`../tutorials/tut_init_files` for more info. 

Argument and Default Value
==========================

Default output file name is `initFile.txt`

Details
=======

Takes the values of the flags from the given dlatkInterface call and creates a text file containing the parameter and it's value. If parameter is equal to the default set in dlaConstants then the value is omitted. 

Below is a list of variables that can be written to a file: 

.. code-block:: bash

	[constants] 
	corpdb = 
	corptable = 
	correl_field = 
	mysql_host = 
	message_field = 
	messageid_field = 
	date_field = 
	lexicondb = 
	feattable = 
	featnames = 
	featlabelmaptable = 
	featlabelmaplex = 
	lextable = 
	wordTable = 
	outcometable = 
	outcomefields = 
	outcomecontrols = 
	outcomeinteraction = 
	groupfreqthresh = 
	outputname = 
	p_correction_method = 
	tagcloudcolorscheme = 
	maxP = 
	encoding = 

Note: when using an init file outside of dlatkInterface, i.e., when using the classes explicitly, most of these values will be ignored. 


Example Commands
================

The command:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 \ 
	-f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' --outcome_table blog_outcomes \ 
	--outcomes is_student --controls age gender --correlate  \ 
	--output_name ~/my_output --p_value 0.05 --p_correction 'bonferroni' --to_file ~/someFile.ini

Creates the file `someFile.ini`:

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
