.. _fwflag_emb_model:
============
--emb_model
============
Switch
======

--add_emb_feat

Description
===========

Adds a Huggingface Transformer feature table.

Argument and Default Value
==========================

Details
=======

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t` , :doc:`fwflag_c`

Optional Switches:

* :doc:`fwflag_emb_layers`
* :doc:`fwflag_emb_msg_aggregation`
* :doc:`fwflag_emb_layer_aggregation` 

Example Commands
================

Creates a default BERT feature table.

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_emb_feat --emb_model bert-base-uncased
