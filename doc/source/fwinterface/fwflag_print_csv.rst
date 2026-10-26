.. _fwflag_print_csv:
===========
--print_csv
===========
Switch
======

--print_csv FILENAME [--print_freq_csv [FILENAME]]

Description
===========

Writes a CSV with one row per group: the group id, its outcomes (prefixed ``outcome_``) and controls (prefixed ``cntrl_``), then one column per feature. By default the feature columns are relative frequencies (``group_norm``); add ``--print_freq_csv`` to write feature counts (``value``) instead.

Argument and Default Value
==========================

FILENAME - the CSV file to write. ``--print_freq_csv`` can also be given its own filename, in which case ``--print_csv`` is not needed.

Other Switches
==============

Required Switches:

* :doc:`fwflag_f`
* :doc:`fwflag_outcome_table` and :doc:`fwflag_outcomes`

Optional Switches:

* --print_freq_csv [FILENAME]
* :doc:`fwflag_group_freq_thresh`

Example Commands
================

.. code-block:: bash

	# relative frequencies (group_norm) of 1grams with age and gender
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$msgs$user_id$0_05' --outcome_table blog_outcomes --outcomes age gender --print_csv users_1grams.csv

	# the same with feature counts (value)
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$msgs$user_id$0_05' --outcome_table blog_outcomes --outcomes age gender --print_csv users_1gram_counts.csv --print_freq_csv
