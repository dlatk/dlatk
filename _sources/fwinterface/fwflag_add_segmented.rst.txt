.. _fwflag_add_segmented:
===============
--add_segmented
===============
Switch
======

--add_segmented

Description
===========

Creates a word-segmented version of the message table (for Chinese only!).

Argument and Default Value
==========================

None

Details
=======

This will create a table called TABLE_seg (where TABLE is specified by :doc:`fwflag_t`) in the database specified by :doc:`fwflag_d`. The message column in this new table is a list of segmented words. Note that word segmentation only means something for Chinese messages.

Choose the segmentation model by using :doc:`fwflag_segmentation_model`. 

After having done this, use :doc:`fwflag_add_ngrams_from_tokenized` to extract ngrams.

How it works:

The infrastructure writes the (message_id, message) pairs to a tempfile, runs the segmentor using the "command line" (os.system) and prints the segmented messages to a different temp file.

The segmentor adds weird things (splits up long numbers; URLS incorrectly), so the python code fixes that.

Weibo by default turns 'emoji' into '[emoji_label_word]' which get's split up by the segmentor, so the python code joins them together again.

Example on one message:

Original message:

.. code-block:: bash

	[神马]欧洲站夏季女装雪纺短袖长裤女士运动时尚休闲套装女夏装2014新款  http://t.cn/RvCypCj

Will turn into:

.. code-block:: bash

	["[\u795e\u9a6c]", "\u6b27\u6d32", "\u7ad9", "\u590f\u5b63", "\u5973\u88c5", "\u96ea\u7eba",
	"\u77ed\u8896", "\u957f\u88e4", "\u5973\u58eb", "\u8fd0\u52a8", "\u65f6\u5c1a", "\u4f11\u95f2",
	"\u5957\u88c5", "\u5973", "\u590f\u88c5", "2014", "\u65b0\u6b3e", "http://t.cn/RvCypCj"]

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 

Optional Switches:

* :doc:`fwflag_segmentation_model` 

Example Commands
================

.. code-block:: bash
	
	# creates the table msgs_seg via the Penn Chinese Treebank
	./dlatkInterface.py -d dla_tutorial -t msgs -c message_id --add_segmented
