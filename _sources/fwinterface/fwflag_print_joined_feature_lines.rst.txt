.. _fwflag_print_joined_feature_lines:
============================
--print_joined_feature_lines
============================
Switch
======

--print_joined_feature_lines

Description
===========

Accomplishes something similar to --print_tokenized_lines, but starting from a feature table instead of a message table. Result is formatted for input to Mallet; see LDA Tutorial.

Argument and Default Value
==========================

If the feature is a multi-word expression joined by spaces, this will replace the spaces with underscores.

Details
=======

This is useful for creating topics from collocations without turning the collocation table back into a message table.

If value > 1, prints that feature [value] times.

If this is run on a non:doc:`fwflag_occurence-filtered` 1:doc:`fwflag_gram` table grouped on message_id, this will act identically to :doc:`fwflag_print_tokenized_lines` (though order of tokens may be different).


Other Switches
==============

Required Switches:
:doc:`fwflag_d`, :doc:`fwflag_t`, :doc:`fwflag_f` Optional Switches:
None

Example Commands
================
.. code:doc:`fwflag_block`:: python


Input feature table:
+----+----------+---------------+-------+--------------------+
| id | group_id | feat          | value | group_norm         |
+----+----------+---------------+-------+--------------------+
|  1 |        1 | a             |     1 | 0.0714285714285714 |
|  2 |        1 | friend        |     1 | 0.0714285714285714 |
|  3 |        1 | off           |     1 | 0.0714285714285714 |
|  4 |        1 | very          |     1 | 0.0714285714285714 |
|  5 |        1 | =)            |     1 | 0.0714285714285714 |
|  6 |        1 | is            |     1 | 0.0714285714285714 |
|  7 |        1 | well          |     1 | 0.0714285714285714 |
|  8 |        1 | .             |     2 |  0.142857142857143 |
|  9 |        1 | to            |     1 | 0.0714285714285714 |
| 10 |        1 | rested        |     1 | 0.0714285714285714 |
| 11 |        1 | starbucks     |     1 | 0.0714285714285714 |
| 12 |        1 | with          |     1 | 0.0714285714285714 |
| 13 |        1 | to catch up   |     1 | 0.0714285714285714 |
| 14 |        2 | and           |     2 | 0.0666666666666667 |
| 15 |        2 | city          |     1 | 0.0333333333333333 |
| 16 |        2 | "             |     4 |  0.133333333333333 |
| 17 |        2 | two           |     1 | 0.0333333333333333 |
| 18 |        2 | .             |     2 | 0.0666666666666667 |
| 19 |        2 | to            |     1 | 0.0333333333333333 |
| 20 |        2 | memory        |     1 | 0.0333333333333333 |
| 21 |        2 | new           |     1 | 0.0333333333333333 |
| 22 |        2 | :             |     1 | 0.0333333333333333 |
| 23 |        2 | excited       |     1 | 0.0333333333333333 |
| 24 |        2 | today         |     1 | 0.0333333333333333 |
| 25 |        2 | soundtrack's  |     1 | 0.0333333333333333 |
| 26 |        2 | got           |     1 | 0.0333333333333333 |
| 27 |        2 | oddessy       |     1 | 0.0333333333333333 |
| 28 |        2 | albums        |     1 | 0.0333333333333333 |
| 29 |        2 | by            |     1 | 0.0333333333333333 |
| 30 |        2 | i am very     |     1 | 0.0333333333333333 |
| 31 |        2 | this          |     1 | 0.0333333333333333 |
| 32 |        2 | motion        |     1 | 0.0333333333333333 |
| 33 |        2 | zombies       |     1 | 0.0333333333333333 |
| 34 |        2 | oracle        |     1 | 0.0333333333333333 |
| 35 |        2 | commit        |     1 | 0.0333333333333333 |
| 36 |        2 | the           |     3 |                0.1 |

Print command:
fwInterface.py :doc:`fwflag_d` fb22 :doc:`fwflag_t` messagesEn :doc:`fwflag_c` message_id :doc:`fwflag_f` 'feat$colloc$messagesEn$message_id$16to16' :doc:`fwflag_print_joined_feature_lines` featCollocLines.txt
Produces in featCollocLines.txt:
1 en a off very with is well . . to rested starbucks to_catch_up =) friend
2 en and and city " " " " two . . to memory got : excited today soundtrack's new 
   oddessy albums by i_am_very this motion zombies oracle commit the the the
