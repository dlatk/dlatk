.. _fwflag_categorical:
============
--categorical
============
Switch
======

--categorical [MySQL column name]

Description
===========

Take a categorical column in MySQL and convert to a one hot representation

Argument and Default Value
==========================

MySQL column name

Details
=======

MySQL column values must be intergers or varchars. For every distinct value it creates a new outcome named after the original outcome with the distinct value appended to the end. For example, if your outcome is "education" with values "highschool", "college" and "phd" then you will have three outcomes: education_highschool, education_college and education_phd. 

If your outcome has two distinct values then only one will be used. For example, if you outcome is "gender" with values "male" and "female" you will get one outcome either "gender__male" or "gender__female". 

The argument must also be listed after either :doc:`fwflag_outcomes` or :doc:`fwflag_outcome_controls`.

Aliases: --categories_to_binary and --cat_to_bin

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 
* :doc:`fwflag_outcomes` or :doc:`fwflag_outcome_controls`

Example Commands
================

Correlate age, gender and occupation groups where gender is a text field containing the values "male" and "female":

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --correlate --csv --outcome_table blog_outcomes --outcomes age gender_cat is_student is_education is_technology --categorical gender_cat --outcome_with_outcome_only --output ~/correlations

