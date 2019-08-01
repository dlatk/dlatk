.. _fwflag_from_file:
===========
--from_file
===========
Switch
======

--from_file FILENAME

Description
===========

Text file where dlatkInterface flags can be set. See also: :doc:`fwflag_to_file`. See :doc:`../tutorials/tut_init_files` for more info. 

Argument and Default Value
==========================

No defaults

Details
=======

Any variable in the text file can be overridden by the command line. Precedence: command line > init file > default. Do NOT quote strings. Lists must be comma separated. 

This is a list of all of the variables that can be read from the file:

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

Given the file *someFile.ini*:

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

the command: 

.. code-block:: bash

	dlatkInterface.py --from_file ~/someFile.ini --correlate

is equivalent to:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --group_freq_thresh 500 \ 
	-f 'feat$cat_met_a30_2000_cp_w$msgs$user_id$16to16' --outcome_table blog_outcomes \ 
	--outcomes is_student --controls age gender --correlate  \ 
	--output_name ~/my_output --p_value 0.05 --p_correction 'bonferroni' --to_file ~/someFile.ini