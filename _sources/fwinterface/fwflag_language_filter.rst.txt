.. _fwflag_language_filter:
=============
--language_filter
=============
Switch
======

--language_filter lang1 [lang2 lang3 ...]

Description
===========

Creates a language filtered message table.

Argument and Default Value
==========================

lang1 is a two letter string identifier for the language we want to filter for. There is no default value.

Details
=======

Uses the `langid <https://github.com/saffsd/langid.py>`_ Python package. By default this will lowercase your messages before running through langid. To turn this off use --no_lower?.

langid is trained on the following languages: af, am, an, ar, as, az, be, bg, bn, br, bs, ca, cs, cy, da, de, dz, el, en, eo, es, et, eu, fa, fi, fo, fr, ga, gl, gu, he, hi, hr, ht, hu, hy, id, is, it, ja, jv, ka, kk, km, kn, ko, ku, ky, la, lb, lo, lt, lv, mg, mk, ml, mn, mr, ms, mt, nb, ne, nl, nn, no, oc, or, pa, pl, ps, pt, qu, ro, ru, rw, se, si, sk, sl, sq, sr, sv, sw, ta, te, th, tl, tr, ug, uk, ur, vi, vo, wa, xh, zh, zu.

:doc:`fwflag_clean_messages` will remove hashtags, URLs and @mentions before processing the message, which will improve classification. 

Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` 

Optional Switches:

* :doc:`fwflag_clean_messages`

Example Commands
================

Remove non English text while cleaning URLs and @mentions:

.. code-block:: bash

	# creates the table msgs_en
	./dlatkInterface.py -d dla_tutorial -t msgs -c user_id --language_filter en --clean_messages

