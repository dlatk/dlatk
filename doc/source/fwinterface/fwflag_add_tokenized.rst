.. _fwflag_add_tokenized:
===============
--add_tokenized
===============
Switch
======

--add_tokenized

Description
===========

Creates a tokenized version of the message table.

Argument and Default Value
==========================

None

Details
=======

This will create a table called TABLE_tok (where TABLE is specified by :doc:`fwflag_t`) in the database specified by :doc:`fwflag_d`. The message column in this new table is a list of tokens. 

This switch is used to create a tokenized version of the message table. It uses WWBP's tokenizer, which splits the message into tokens, then dumps the JSON version of the token list into MySQL text.
If your message is:
 "Mom said she's gonna think about getting a truck."
the same row in the tokenized table will look like this:
 ["mom", "said", "she's", "gonna", "think", "about", "getting", "a", "truck", "."]
To use the tokenized table in standalone scripts, simply do JSON.load(message).


Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` Optional Switches:
:doc:`fwflag_message_field` <FIELD> 
None

Example Commands
================
.. code:doc:`fwflag_block`:: python


 # General form
 # Creates the tables: TABLE_tok
 ./fwInterface.py :doc:`fwflag_d` DATABASE :doc:`fwflag_t` TABLE :doc:`fwflag_c` GROUP_BY_FIELD :doc:`fwflag_add_tokenized` 
 # Creates the tables: primals_tok
 ./fwInterface.py :doc:`fwflag_d` primals :doc:`fwflag_t` primals_new :doc:`fwflag_c` message_id :doc:`fwflag_add_tokenized` 