.. _fwflag_bert_layers:
=============
--bert_layers
=============
Switch
======

--bert_layers N [N1 N2 ... Nk]

Description
===========

Specifies the BERT layers to aggregate over. One or more layers may be specified.

Argument and Default Value
==========================

Default: the last four layers (8-11).

Details
=======

Layers should be specified by index, beginning from the 0th index. 11 is the index of the final (12th) layer.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t` , :doc:`fwflag_c`

Optional Switches:

* :doc:`fwflag_bert_model`
* :doc:`fwflag_bert_msg_aggregation`
* :doc:`fwflag_bert_layer_aggregation` 

Example Commands
================

Creates BERT features aggregated over the last two layers.

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_bert --bert_layers 10 11
