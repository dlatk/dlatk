.. _fwflag_tf_idf:
========
--tf_idf
========
Switch
======

--tf_idf

Description
===========

Creates new feature table where the group_norm is the tf-idf score. Each group_id is seen as a document for calculating tf-idf.

Argument and Default Value
==========================

None

Details
=======

The :doc:`fwflag_f` flag should be an ngram table.

Resulting value refers to value in ngram table. Group_norm refers to tf:doc:`fwflag_idf` score.


Other Switches
==============

Required: 

* :doc:`fwflag_d`, :doc:`fwflag_t`, :doc:`fwflag_c` 
* :doc:`fwflag_f` 

Example Commands
================

.. code-block:: bash

   ./dlatkInterface.py -d dla_tutorial -t msgs -c user_id -f 'feat$1gram$msgs$user_id$16to16' --tf_idf

.. code-block:: mysql

   mysql> select * from feat$tf_idf_1gram$msgs$user_id order limit 5;;
   +---------+----------+-----------+-------+--------------------+
   | id      | group_id | feat      | value | group_norm         |
   +---------+----------+-----------+-------+--------------------+
   |  307349 |  2033616 | delivered |     1 | 0.0000878334772103 |
   |  278647 |  4144593 | crap      |     6 |  0.000998442620366 |
   | 1043863 |  3482840 | story     |     2 |  0.000334689956064 |
   | 1150911 |  2876677 | uh        |     2 |  0.000141436336165 |
   |  283547 |  3711805 | crosses   |     2 |  0.000827587016091 |
   +---------+----------+-----------+-------+--------------------+ 