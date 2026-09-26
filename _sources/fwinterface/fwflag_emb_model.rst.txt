.. _fwflag_emb_model:
============
--emb_model
============
Switch
======

--emb_model model_name

Description
===========

The name of the Transformer model to use from `Hugging Face <https://huggingface.co/models>`.

Argument and Default Value
==========================

Default: `bert-base-uncased`

Details
=======

Specify a HuggingFace model to use. 

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

Creates a BERT feature table using ``bert-large-uncased``

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_emb_feat --emb_model bert-large-uncased
