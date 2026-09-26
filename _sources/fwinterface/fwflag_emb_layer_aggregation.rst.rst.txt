.. _fwflag_emb_layer_aggregation:
========================
--emb_layer_aggregation
========================
Switch
======

--emb_layer_aggregation aggregation_method

Description
===========

Specifies the aggregation method to apply over transformer layers to produce a single vector representation. The layers to aggregate over are specified with :doc:`fwflag_emb_layers`.

Argument and Default Value
==========================

Default: concatenate. Currently accepts only one aggregation method.

Details
=======

This option is interpreted as a numpy method name, applied to the 0th axis.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t` , :doc:`fwflag_c`

Optional Switches:

* :doc:`fwflag_emb_model`
* :doc:`fwflag_emb_msg_aggregation`
* :doc:`fwflag_emb_layers` 

Example Commands
================

Creates a Transformers feature table summed over the last two layers.

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_emb_feat --emb_model bert-base-uncased --emb_layers 11 12 --emb_layer_aggregation sum
