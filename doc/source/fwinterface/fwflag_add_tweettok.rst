.. _fwflag_add_tweettok:
==============
--add_tweettok
==============
Switch
======

--add_tweettok

Description
===========

Creates a tokenized, by TweetNLP, version of the message table.

Argument and Default Value
==========================

None

Details
=======

This will create a table called TABLE_tweettok (where TABLE is specified by :doc:`fwflag_t`) in the database specified by :doc:`fwflag_d`. The message column in this new table is a list of tokens. 

Example on one message
Original message:
 "@antijokeapple: What do you call a Bee who is having a bad hair day? A Frisbee." Hahah. 
Tokenized message:
 ["\"", "@antijokeapple", ":", "What", "do", "you", "call", "a", "Bee", "who", "is", "having", "a", 
 "bad", "hair", "day", "?", "A", "Frisbee", ".", "\"", "Hahah", "."]

Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t` Optional Switches:
:doc:`fwflag_messageid_field` :doc:`fwflag_message_field` 
Example Commands
================
.. code:doc:`fwflag_block`:: python


 # General form
 # Creates the tables: TABLE_tok
 ./fwInterface.py :doc:`fwflag_d` DATABASE :doc:`fwflag_t` TABLE :doc:`fwflag_c` GROUP_BY_FIELD :doc:`fwflag_add_tweettok` 