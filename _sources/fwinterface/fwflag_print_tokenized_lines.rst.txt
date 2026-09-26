.. _fwflag_print_tokenized_lines:
=======================
--print_tokenized_lines
=======================
Switch
======

--print_tokenized_lines

Description
===========

Prints tokenized version of messages to lines.

Argument and Default Value
==========================

You must supply an output file name.

Details
=======

Looks for the table TABLENAME_tok, where TABLENAME is specified by :doc:`fwflag_t`. 
Each line of the output file contains the message id, lanugage, and tokens. Example:
 # Sample message from tokenized input table:
 # ["is", "worth", "it", "just", "follow", "your", "heart", "its", "never", "wrong", ":", "-", "rrb", "-"]
 # Output line:
 # 128675651556356096 en is worth it just follow your heart its never wrong : - rrb -

Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_t` Optional Switches:
:doc:`fwflag_feat_whitelist` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # General command
 python fwInterface.py :doc:`fwflag_d` DATABASE :doc:`fwflag_t` TABLE :doc:`fwflag_print_tokenized_lines` OUTPUTFILE_NAME

 # Example command
 # searches for the table twt_20mil_tok
 # outputs the file twt_20mil.txt
 python fwInterface.py :doc:`fwflag_d` twitterGH :doc:`fwflag_t` twt_20mil :doc:`fwflag_print_tokenized_lines` twt_20mil.txt
