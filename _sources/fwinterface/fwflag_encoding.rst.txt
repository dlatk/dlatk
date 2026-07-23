.. _fwflag_encoding:
==========
--encoding
==========
Switch
======

--encoding ENCODING_TYPE

Description
===========

Specify the type of encoding used when interacting with MySQL.

Argument and Default Value
==========================

Default: utf8mb4

Details
=======

Possible encodings (with collations):

.. code-block:: bash

	utf8mb4 (utf8mb4_bin)
	utf8 (utf8_general_ci)
	latin1 (latin1_swedish_ci) 
	latin2 (latin2_general_ci)
	ascii (ascii_general_ci)

Example Commands
================

.. code-block:: bash

	dlatkInterface.py -d dla_tutorial -t msgs_xxx -c user_id --add_ngrams -n 1 2 3 --encoding latin1