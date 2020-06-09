.. _fwflag_bert_no_context:
=================
--bert_no_context
=================
Switch
======

--bert_no_context

Description
===========



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

Creates 1gram features and save in `feat$1gram$msgs$user_id$16to16`.

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_ngrams


This command will create tables for 1grams (`feat$1gram$msgs$user_id$16to16`) and 2grams (`feat$2gram$msgs$user_id$16to16`) and then a third table which contains 1-2grams (`feat$1to2gram$msgs$user_id$16to16`). 

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs -c user_id --add_ngrams -n 1 2 --combine_feat_tables 1to2gram



