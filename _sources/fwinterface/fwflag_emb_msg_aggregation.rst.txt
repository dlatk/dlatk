.. _fwflag_emb_msg_aggregation:
======================
--emb_msg_aggregation
======================
Switch
======

--emb_msg_aggregation aggregation_method

Description
===========

Specify the aggregation function to apply over embedded messages to produce a single embedding of the group_id (:doc:`fwflag_c`).

Argument and Default Value
==========================

Default: mean.

Details
=======

Any method that can be called as `np.method(bertMessageVectors, axis=0)`.

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_t` , :doc:`fwflag_c`

Optional Switches:

* :doc:`fwflag_emb_model`
* :doc:`fwflag_emb_layer_aggregation`
* :doc:`fwflag_emb_layers` 

Example Commands
================

Creates a BERT feature table with messages aggregated by selecting the median.

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_emb_feat --emb_model bert-large-uncased --emb_msg_aggregation median
