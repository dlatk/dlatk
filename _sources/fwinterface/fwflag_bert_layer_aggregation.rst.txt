.. _fwflag_bert_layer_aggregation:
========================
--bert_layer_aggregation
========================
Switch
======

--bert_layer_aggregation aggregation_method

Description
===========

Specifies the aggregation method to apply over bert layers to produce a single vector representation. The layers to aggregate over are specified with :doc:`fwflag_bert_layers`.

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

* :doc:`fwflag_bert_model`
* :doc:`fwflag_bert_msg_aggregation`
* :doc:`fwflag_bert_layers` 
* :doc:`fwflag_bert_word_aggregation` 
* :doc:`fwflag_bert_no_context` 

Example Commands
================

Creates a BERT feature table summed over the last two layers.

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_bert --bert_layers 10 11 --bert_layer_aggregation sum
