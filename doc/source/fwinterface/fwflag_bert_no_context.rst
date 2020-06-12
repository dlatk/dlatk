.. _fwflag_bert_no_context:
=================
--bert_no_context
=================
Switch
======

--bert_no_context

Description
===========

Disables contextual embedding. This means words will be embedded individually, rather than in each others context.

Argument and Default Value
==========================



Details
=======



Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t` , :doc:`fwflag_c`

Optional Switches:

* :doc:`fwflag_bert_model`
* :doc:`fwflag_bert_layer_aggregation`
* :doc:`fwflag_bert_msg_aggregation`
* :doc:`fwflag_bert_layers` 
* :doc:`fwflag_bert_word_aggregation` 


Example Commands
================

Creates a BERT feature table in which words are embedded without their context.

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_bert --no_context
