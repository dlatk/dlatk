.. _fwflag_cca_penalty_outcomes:
======================
--cca_penalty_outcomes
======================
Switch
======

--cca_penalty_outcomes, --cca_penatlyz

Description
===========

Sets the penalty for the Z matrix during CCA

Argument and Default Value
==========================

Penalty, between 0 and 1; default is None

Details
=======

Sets sparsity penalty for the outcomees (and controls). A higher number usually means less penalization.
Do :doc:`fwflag_cca_permute` to get an idea of what a good value is.


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`
* :doc:`fwflag_f`
* :doc:`fwflag_outcome_table`, :doc:`fwflag_outcomes`
* :doc:`fwflag_cca` 

Example Commands
================

.. code-block:: bash


see :doc:`fwflag_cca` 
