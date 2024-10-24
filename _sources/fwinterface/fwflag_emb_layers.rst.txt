.. _fwflag_emb_layers:
=============
--emb_layers
=============
Switch
======

--emb_layers N [N1 N2 ... Nk]

Description
===========

Specifies the Transformers layers to aggregate over. One or more layers may be specified.

Argument and Default Value
==========================

Default: the last four layers (9-12).

Details
=======

Layers should be specified by index, 0th index would be static word representations on which contextual embeddings are computed upon and 1st through 12th index for first to final (12th) layer respectively of a bert-base model.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t` , :doc:`fwflag_c`

Optional Switches:

* :doc:`fwflag_emb_model`
* :doc:`fwflag_emb_msg_aggregation`
* :doc:`fwflag_emb_layer_aggregation` 

Example Commands
================

Creates BERT features aggregated over the last two layers.

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_emb_feat --emb_model bert-base-uncased --emb_layers 11 12
