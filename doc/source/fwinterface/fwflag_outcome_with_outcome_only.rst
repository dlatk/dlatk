.. _fwflag_outcome_with_outcome_only:
===========================
--outcome_with_outcome_only
===========================
Switch
======

--outcome_with_outcome_only

Description
===========

Correlate a list of outcomes with each other

Argument and Default Value
==========================

None, default is false.

Details
=======

Similar to :doc:`fwflag_outcome_with_outcome` except no language features are considered. 

If a feature table is given then :doc:`fwflag_group_freq_thresh` will be applied, otherwise we consider all users in the outcome table.


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t`, :doc:`fwflag_c` 
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes` 

Optional Switches:

* :doc:`fwflag_f` 
* :doc:`fwflag_group_freq_thresh`

Example Commands
================

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --correlate --csv \
	--outcome_table blog_outcomes --outcomes age gender is_student is_education is_technology \
	--outcome_with_outcome_only


