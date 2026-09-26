.. _fwflag_sparse:
========
--sparse
========
Switch
======

--sparse

Description
===========

Use sparse representation for X when training / testing.

Argument and Default Value
==========================

Default value is False.

Details
=======

Often calls the Scipy csr_matrix (Compressed Sparse Row) class. Sparse matrices can be used in arithmetic operations: they support addition, subtraction, multiplication, division, and matrix power.

Advantages of the CSR format
efficient arithmetic operations CSR + CSR, CSR * CSR, etc.
efficient row slicing
fast matrix vector products
Disadvantages of the CSR format
slow column slicing operations
changes to the sparsity structure are expensive 

Other Switches
==============

Optional Switches:
:doc:`fwflag_train_regression`, :doc:`fwflag_train_reg` :doc:`fwflag_test_regression` :doc:`fwflag_combo_test_regression`, :doc:`fwflag_combo_test_reg` :doc:`fwflag_control_adjust_outcomes_regression`, :doc:`fwflag_control_adjust_reg`? :doc:`fwflag_test_combined_regression`? :doc:`fwflag_predict_regression`, :doc:`fwflag_predict_reg` :doc:`fwflag_predict_regression_to_feats` :doc:`fwflag_predict_cv_to_feats`, :doc:`fwflag_predict_combo_to_feats`, :doc:`fwflag_predict_regression_all_to_feats`? :doc:`fwflag_train_classifiers`, :doc:`fwflag_train_class` :doc:`fwflag_test_classifiers` :doc:`fwflag_combo_test_classifiers` :doc:`fwflag_predict_classifiers`, :doc:`fwflag_predict_class` :doc:`fwflag_roc` :doc:`fwflag_predict_classifiers_to_feats` :doc:`fwflag_predict_cv_to_feats`, :doc:`fwflag_predict_combo_to_feats`, :doc:`fwflag_predict_regression_all_to_feats`? :doc:`fwflag_train_c2r`? :doc:`fwflag_test_c2r`? :doc:`fwflag_predict_c2r`? :doc:`fwflag_fit_reducer` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


