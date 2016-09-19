.. _fwflag_segmentation_model:
====================
--segmentation_model
====================
Switch
======

--segmentation_model

Description
===========

Choose the model for message segmentation

Argument and Default Value
==========================

ctb

Details
=======

Chooses which trained model to use for :doc:`fwflag_add_segmented`. 

Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` :doc:`fwflag_add_segmented` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # Uses CBT model to segment the messages in messages_2014_06 and put them into
 # a new table called messages_2014_06_seg
 ~/fwInterface.py :doc:`fwflag_d` randomWeibo :doc:`fwflag_t` messages_2014_06 :doc:`fwflag_c` message_id :doc:`fwflag_add_segmented` :doc:`fwflag_segmentation_model` ctb
