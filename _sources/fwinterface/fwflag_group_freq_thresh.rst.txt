.. _fwflag_group_freq_thresh:
===================
--group_freq_thresh
===================
Switch
======

--group_freq_thresh

Description
===========

This is one of the more important flags within DLATK. Minimum WORD frequency per correl_field to include correl_field in results.

Argument and Default Value
==========================

Argument is an integer number. Unless specified the default value is set based on the the follow rules (based on the argument to the :doc:`fwflag_c` flag):

.. code-block:: python

    if any(field in correl_field.lower() for field in ["mess", "msg"]) or correl_field.lower().startswith("id"):
        group_freq_thresh = 1
    elif any(field in correl_field.lower() for field in ["user", "usr", "auth"]):
        group_freq_thresh = 500
    elif any(field in correl_field.lower() for field in ["county", "cnty", "cty", "fips"]):
        group_freq_thresh = 40000
    else:
    	group_freq_thresh = 500


Details
=======

Counts the number of words in each group specified by :doc:`fwflag_c` (the correl or group field). If this count is less than the given group frequency threshold then this group is thrown out. The group is otherwise kept. 

Specifically it will query the `word table` which by default is your 1gram table. In MySQL this table name is pieced together from the :doc:`fwflag_d`, :doc:`fwflag_t` and :doc:`fwflag_c` flags. So for example if your base command is

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id

then MySQL will query the table `feat$1gram$msgs$user_id$16to16`. It will sum the `value` column while grouping by `group_id` and throw away any `group_id` with a sum less than the specified group frequency threshold. 

Other important details:

* Non-default words tables can be specified using the :doc:`fwflag_word_table` flag. 
* If the group frequency threshold is set to zero then groups are queried from your message table (:doc:`fwflag_t`).
* This flag is used in all non-feature extraction type DLATK commands, except for feature occurrence filtering (:doc:`../fwinterface/fwflag_feat_occ_filter`). 
* For document (or message, sentence, very short text, etc.) analysis setting a group frequency threshold greater than zero can cause the MySQL server to time out. As a work around you can make sure no documents are empty and then set `--group_freq_thresh 0`. 


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t`, :doc:`fwflag_c`

Example Commands
================

This first example we subset the feature table to only those features used by 5% of groups, when considering only groups with 500 words or more. This creates the table `feat$1to3gram$msgs$user_id$16to16$0_05`.

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1to3gram$msgs$user_id$16to16' \
	--feat_occ_filter --set_p_occ 0.05 --group_freq_thresh 500

In this example we correlate age and gender with 1to3grams, only considering users with 500 or more words:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id \ 
	-f 'feat$1to3gram$msgs$user_id$16to16$0_05' \ 
	--outcome_table blog_outcomes  --group_freq_thresh 500 \ 
	--outcomes age gender --output_name xxx_output \ 
	--tagcloud --make_wordclouds
