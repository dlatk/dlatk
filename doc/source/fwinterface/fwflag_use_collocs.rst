.. _fwflag_use_collocs:
=============
--use_collocs
=============
Switch
======

--use_collocs

Description
===========

Use a set of collocations to extract n grams.

Argument and Default Value
==========================

Use this option to extract features using a collocation table (--colloc_table), or to modify a feature table that was extracted using collocations.  The collocation table holds the multigrams that should be considered together.  All words that aren’t part of the predefined list of collocations will be counted as 1grams.

Details
=======

Use this option to extract features using a collocation table (:doc:`fwflag_colloc_table`), or to modify a feature table that was extracted using collocations.  The collocation table holds the multigrams that should be considered together.  All words that aren’t part of the predefined list of collocations will be counted as 1grams.  

Note: :doc:`fwflag_colloc_table` is assumed to have columns ‘feat’

Note: The preferred collocation table as of June 2015 is ufeat$pmi$fb22_messagesEn$lnpmi0_15


Other Switches
==============

Required Switches:

None

Optional Switches:

* :doc:`fwflag_colloc_table` <TABLENAME>
* :doc:`fwflag_include_sub_collocs` :doc:`fwflag_feature_type_name` <STRING>

Example Commands
================

.. code-block:: bash

	# Extract and filter in one command
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_ngrams --use_collocs --colloc_table 'ufeat$pmi$msgs$lnpmi0_15' --feat_occ_filter --set_p_occ 0.05



	# Add a filter to a table that was generated using collocs, (requires specifying the word table for group_frequency calculation)
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$colloc$msgs$user_id$16to16’ --word_table ’feat$colloc$msgs$user_id$16to16’ --feat_occ_filter --set_p_occ 0.05

Example outputs: 

* feat$colloc$msgsEn_r5k$user_id$16to16
* feat$colloc$msgsEn_r5k$user_id$16to16$0_05

