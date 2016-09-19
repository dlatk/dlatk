.. _fwflag_mediation_boot:
================
--mediation_boot
================
Switch
======

--mediation_boot

Description
===========

To be used in Mediation analysis to specify a bootstrapping method during the significance test.  This has a default number of 1000 repetitions. A parametric method is used as the default significance testing method.

Argument and Default Value
==========================

Required Switches:

Details
=======

To be used in Mediation analysis to specify a bootstrapping method during the significance test.  This has a default number of 1000 repetitions. A parametric method is used as the default significance testing method.


Other Switches
==============

Required Switches:
:doc:`fwflag_mediation` Optional Switches:
:doc:`fwflag_mediation_boot_num` Note: there are many other required and optional switches when running :doc:`fwflag_mediation`. 

Example Commands
================
.. code:doc:`fwflag_block`:: python


 ./fwInterface.py :doc:`fwflag_d` twitterGH :doc:`fwflag_t` messages_en :doc:`fwflag_c` cty_id :doc:`fwflag_f` 'feat$cat_met_a30_2000_cp_w$messages_en$cty_id$16to16' :doc:`fwflag_outcome_table` nejm_intersect_small50k \ 
 :doc:`fwflag_mediation` :doc:`fwflag_path_start` 'hsgradHC03_VC93ACS3yr$10'  :doc:`fwflag_outcomes` 'ucd_I25_1_atheroHD$0910_ageadj' :doc:`fwflag_mediators`  '1051' :doc:`fwflag_mediation_boot` :doc:`fwflag_mediation_boot_num` 10000
