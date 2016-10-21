.. _fwflag_add_tweetpos:
==============
--add_tweetpos
==============
Switch
======

--add_tweetpos

Description
===========

Creates a POS tagged, by TweetNLP version of the message table.

Argument and Default Value
==========================

None

Details
=======

This will create a table called TABLE_tweetpos (where TABLE is specified by :doc:`fwflag_t`) in the database specified by :doc:`fwflag_d`. The message column in this new table is a list of tokens. 

Example on one message
Original message:
 @f_ckj i think that curly hair is getting to you 😂
POS message:
 {"tokens": ["@f_ckj", "i", "think", "that", "curly", "hair", "is", "getting", "to", "you", "&", "#128514", ";"], 
 "original": "@f_ckj i think that curly hair is getting to you 😂", 
 "probs": ["0.9994", "0.9898", "0.9999", "0.4810", "0.9903", "0.9992", "0.9955", "0.9959", "0.9967", "0.9992", "0.9806", "0.3757", "0.9448"], 
 "tags": ["@", "O", "V", "D", "A", "N", "V", "V", "P", "O", "&", "#", ","]} 

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
 ./fwInterface.py :doc:`fwflag_d` DATABASE :doc:`fwflag_t` TABLE :doc:`fwflag_c` GROUP_BY_FIELD :doc:`fwflag_add_tweetpos` 