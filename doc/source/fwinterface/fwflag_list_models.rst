.. _fwflag_list_models:
=============
--list_models
=============
Switch
======

--list_models

Description
===========

Lists the models that :doc:`fwflag_model` accepts, then exits. Regression models come first, then classification models, each grouped by type (linear, kernel support vector machine, tree ensemble, naive Bayes, neural network). Each model shows its scikit-learn class and the settings dlatk uses (bracketed lists are searched by cross-validation). The defaults are marked with ``*``.

The list is read from the regression and classification predictors, so it is always current. No database is needed.

Argument and Default Value
==========================

None.

Other Switches
==============

Related Switches:

* :doc:`fwflag_model`
* :doc:`fwflag_combo_test_regression`
* :doc:`fwflag_combo_test_classifiers`

Example Commands
================

.. code-block:: bash

	dlatkInterface.py --list_models
