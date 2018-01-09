.. _fwflag_whitelist:
===========
--whitelist
===========
Switch
======

--whitelist

Description
===========

Switch to enable whitelisting of features in correlation-type analyses.

Argument and Default Value
==========================

None

Details
=======

Requires :doc:`fwflag_feat_whitelist`. Argument of that switch gives either feature table as a whitelist (distinct features of that table) or arguments are a list of features to whitelist.

See also blacklist and feat_blacklist


Other Switches
==============

Required Switches:

* :doc:`fwflag_feat_whitelist` 

Example Commands
================

.. code-block:: bash

	# Only correlate with features agr and con:
	dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$p_ridg_lbp_ocean$msgs$user_id' --group_freq_thresh 0 --outcome_table blog_outcomes --outcomes age  --controls gender --correlate --rmatrix --whitelist --feat_whitelist agr con --output_name ~/my_output



