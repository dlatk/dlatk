.. _fwflag_bert_model:
============
--bert_model
============
Switch
======

--bert_model model_name

Description
===========

The name of the BERT model to use from `Hugging Face <https://huggingface.co/models>`.

Argument and Default Value
==========================

Default: `base-uncased`

Details
=======

Specify a Hugging Face model to use. You can specify "base-*" or "large-*" for BERT models; non-BERT models may also work, but they have not been tested fully.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t` , :doc:`fwflag_c`

Optional Switches:

* :doc:`fwflag_bert_layers`
* :doc:`fwflag_bert_msg_aggregation`
* :doc:`fwflag_bert_layer_aggregation` 

Example Commands
================

Creates a BERT feature table using ``bert-large-uncased``

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_bert --bert_model large-uncased
