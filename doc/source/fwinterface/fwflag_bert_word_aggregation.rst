.. _fwflag_bert_word_aggregation:
=======================
--bert_word_aggregation
=======================
Switch
======

--bert_word_aggregation

Description
===========

Specifies a method to aggregate over words to produce a single message embedding.

Argument and Default Value
==========================

Default: mean.

Details
=======

Any numpy method that can be called as `np.method(sentEncPerWord, axis=0)`.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t` , :doc:`fwflag_c`

Optional Switches:

* :doc:`fwflag_bert_model`
* :doc:`fwflag_bert_layer_aggregation`
* :doc:`fwflag_bert_msg_aggregation`
* :doc:`fwflag_bert_layers` 
* :doc:`fwflag_bert_no_context` 


Example Commands
================

Creates a BERT feature table that aggregates words by summing over their vectors.

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_bert --bert_word_aggregation sum
