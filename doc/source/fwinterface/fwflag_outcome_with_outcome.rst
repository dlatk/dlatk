.. _fwflag_outcome_with_outcome:
======================
--outcome_with_outcome
======================
Switch
======

--outcome_with_outcome

Description
===========

Adds the outcomes themselves to the list of variables to correlate with the outcomes.

Argument and Default Value
==========================

None, default is false.

Details
=======

When doing correlation analysis (DLA) using :doc:`fwflag_rmatrix`, :doc:`fwflag_correlate` or :doc:`fwflag_tagcloud`, this appends the outcomes to the list of features to be correlated with the outcomes.

This means that the output (rmatrix or other) will have extra lines prefixed with outcome_.


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t`, :doc:`fwflag_c` 
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` 
* :doc:`fwflag_f`

Optional Switches:

* :doc:`fwflag_group_freq_thresh`

Example Commands
================

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id  \
	-f 'feat$1gram$msgs$16to16' --correlate --csv \
	--outcome_table blog_outcomes --outcomes age gender is_student is_education is_technology \
	--outcome_with_outcome 