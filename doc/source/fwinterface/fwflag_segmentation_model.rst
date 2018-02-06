.. _fwflag_segmentation_model:
====================
--segmentation_model
====================
Switch
======

--segmentation_model model

Description
===========

Choose the model for message segmentation: ctb (default, Penn Chinese Treebank) or pku (Beijing University)

Argument and Default Value
==========================

ctb

Details
=======

Chooses which trained model to use for :doc:`fwflag_add_segmented`. 

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 
* :doc:`fwflag_add_segmented` 

Example Commands
================

Use the Penn Chinese Treebank:

.. code-block:: bash
	
	# ctb is the default
	./dlatkInterface.py -d dla_tutorial -t msgs -c message_id --add_segmented  

	# or set explicitly 
	./dlatkInterface.py -d dla_tutorial -t msgs -c message_id --add_segmented --segmentation_model cbt

Use the Beijing University model:

.. code-block:: bash
	
	./dlatkInterface.py -d dla_tutorial -t msgs -c message_id --add_segmented  --segmentation_model pku