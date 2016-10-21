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
utf8mb4 (utf8mb4_bin)
utf8 (utf8_general_ci)
latin1 (latin1_swedish_ci) 
latin2 (latin2_general_ci)
ascii (ascii_general_ci)

Example Commands
================
.. code:doc:`fwflag_block`:: python


 ./fwInterface.py :doc:`fwflag_d` dla_tutorial :doc:`fwflag_t` msgs_xxx :doc:`fwflag_c` user_id :doc:`fwflag_add_ngrams` :doc:`fwflag_n` 1 2 3 :doc:`fwflag_encoding` latin1