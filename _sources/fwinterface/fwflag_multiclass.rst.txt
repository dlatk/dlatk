.. _fwflag_multiclass:
============
--multiclass
============
Switch
======

--multiclass [MySQL column name(s)]

Description
===========

Take a multiclass column in MySQL and convert to an integer representation

Argument and Default Value
==========================

MySQL column name(s)

Details
=======

MySQL column values must be varchars or text. For every distinct value in the MySQL column we map it to an integer and create a new outcome named after the original with "__multiclass" appended to the end. For example, if your outcome is "education" with values "highschool", "college" and "phd" then you will have the outcome "education__multiclass": "college" = 0, "high school" = 1 and "phd" = 2. Note that the integers start with 0 and the strings are mapped in alphabetical order. 

The argument must also be listed after either :doc:`fwflag_outcomes` or :doc:`fwflag_outcome_controls`.

Aliases: --categories_to_integer and --cat_to_int

Advanced: The mapping is stored in the `OutcomeGetter` object under the parameter `multiclass_outcome` after running the `getGroupsAndOutcomes` method. 

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 
* :doc:`fwflag_outcomes` or :doc:`fwflag_outcome_controls`

Example Commands
================

Correlate age, gender and sign groups where sign is a text field containing the user's astrological sign:

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --correlate --csv --outcome_table blog_outcomes --outcomes age gender_cat sign --multiclass sign --outcome_with_outcome_only --output ~/correlations

This column will be mapped to:

.. code-block:: python

	{'sign__multiclass': {'cancer': 0, 'libra': 4, 'capricorn': 1, 'taurus': 7, 'gemini': 2, 'pisces': 5, 'scorpio': 6, 'leo': 3}}
