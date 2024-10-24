.. _fwflag_aggregate_feats_by_new_group:
==============================
--aggregate_feats_by_new_group
==============================
Switch
======

--aggregate_feats_by_new_group

Description
===========

Aggregate feature table by group field (i.e. message_id features by user_ids).

Argument and Default Value
==========================

Required Switches:

Details
=======


Other Switches
==============

Required Switches:

* :doc:`fwflag_d`, :doc:`fwflag_c`, :doc:`fwflag_t`, :doc:`fwflag_f` 

Optional Switches:

* None

Example Commands
================

Example use case: feature tables broken down by month being recombined.
Feature tables in question:

.. code-block:: bash

	| feat$1to3gram$msgsPA_2012$cntyYM$16to16$0_01                   |
	| feat$1to3gram$msgsPA_2012_01$cnty$16to16$0_01                  |
	| feat$1to3gram$msgsPA_2012_02$cnty$16to16$0_01                  |
	| feat$1to3gram$msgsPA_2012_03$cnty$16to16$0_01                  |
	| feat$1to3gram$msgsPA_2012_04$cnty$16to16$0_01                  |
	| feat$1to3gram$msgsPA_2012_05$cnty$16to16$0_01                  |
	| feat$1to3gram$msgsPA_2012_06$cnty$16to16$0_01                  |
	| feat$1to3gram$msgsPA_2012_07$cnty$16to16$0_01                  |
	| feat$1to3gram$msgsPA_2012_08$cnty$16to16$0_01                  |
	| feat$1to3gram$msgsPA_2012_09$cnty$16to16$0_01                  |
	| feat$1to3gram$msgsPA_2012_10$cnty$16to16$0_01                  |
	| feat$1to3gram$msgsPA_2012_11$cnty$16to16$0_01                  |
	| feat$1to3gram$msgsPA_2012_12$cnty$16to16$0_01                  |

Note that feat$1to3gram$msgsPA_2012$cntyYM$16to16$0_01 has been created to hold the intermediate step.

Original tables like feat$1to3gram$msgsPA_2012_12$cnty$16to16$0_01:

.. code-block:: bash

	+----------+--------------+-------+--------------------------+
	| group_id | feat         | value | group_norm               |
	+----------+--------------+-------+--------------------------+
	|    42101 | soon as      |    73 |  0.000037435667062561665 |
	|    42017 | are now      |     1 |  0.000019275622120703946 |
	|    42077 | i tell u     |     1 |  0.000036302911493501776 |
	|    42003 | fuck out .   |     3 |  0.000005261560964829973 |
	|    42025 | the last day |     1 |   0.00040225261464199515 |
	|    42101 | it ! we      |     2 | 0.0000011117948649530572 |
	|    42017 | is willing   |     1 |  0.000019275622120703946 |
	|    42049 | kno wat      |     1 |   0.00003332777870354941 |
	|    42003 | just barely  |     1 | 0.0000016047294586605644 |
	|    42101 | 2006         |    48 |   0.00002283123484160593 |
	+----------+--------------+-------+--------------------------+

Initial collapse mysql command, e.g. (note different group field, cntyYM instead of cnty):

.. code-block:: bash
	insert into feat$1to3gram$msgsPA_2013$cntyYM$16to16$0_01
	  select concat(group_id,'_2013_01') group_id,feat,value,group_norm
	  from feat$1to3gram$msgsPA_2013_01$cnty$16to16$0_01;

First collapsed table (feat$1to3gram$msgsPA_2012$cntyYM$16to16$0_01):

.. code-block:: bash

	+---------------+------------------+-------+--------------------------+
	| group_id      | feat             | value | group_norm               |
	+---------------+------------------+-------+--------------------------+
	| 42091_2012_11 | #theatreproblems |     1 |   0.00001484692817056151 |
	| 42101_2012_02 | inappropriate    |     7 | 0.0000026320123388738445 |
	| 42125_2012_03 | for everything   |     1 |  0.000040645449741901396 |
	| 42101_2012_05 | fun and          |    18 |  0.000013396794444795903 |
	| 42071_2012_02 | at my desk       |     1 |  0.000019890601690701143 |
	| 42071_2012_11 | album will       |     1 |  0.000019987607683236393 |
	| 42003_2012_06 | waste it         |     1 | 0.0000012686121256484195 |
	| 42003_2012_08 | tweet goes out   |     1 |  0.000006096408605690388 |
	| 42101_2012_04 | day :            |    29 |    0.0000182426537148019 |
	| 42101_2012_07 | girl you better  |     1 | 0.0000007502882982786135 |
	+---------------+------------------+-------+--------------------------+

Create a new dummy "message table" containing mappings from new group_id to original group_id with mysql commands like:
insert into cntyYM_to_cnty values('42007_2012_01','42007');

Resulting table (cntyYM_to_cnty):

.. code-block:: bash

	+---------------+-------+
	| cntyYM        | cnty  |
	+---------------+-------+
	| 42001_2012_01 | 42001 |
	| 42001_2012_02 | 42001 |
	| 42001_2012_03 | 42001 |
	| 42001_2012_04 | 42001 |
	| 42001_2012_05 | 42001 |
	+---------------+-------+

Aggregation fwInterface command:

.. code-block:: bash

	dlatkInterface.py -d paHealth -t cntyYM_to_cnty -c cnty \
	  -f 'feat$1to3gram$msgsPA_2012$cntyYM$16to16$0_01' \
	  --aggregate_feats_by_new_group

At its heart, this function runs two SQL commands:

.. code-block:: bash

	INSERT INTO feat$agg_1to3gram$msgsPA_2012$cnty
	  SELECT m.cnty, f.feat, sum(f.value), 0 FROM feat$1to3gram$msgsPA_2012$cntyYM$16to16$0_01 AS f,
	  cntyYM_to_cnty AS m where m.cntyYM = f.group_id GROUP BY m.cnty, f.feat
	UPDATE feat$agg_1to3gram$msgsPA_2012$cnty a INNER JOIN
	  (SELECT group_id,sum(value) sum FROM feat$agg_1to3gram$msgsPA_2012$cnty
	  GROUP BY group_id) b ON a.group_id=b.group_id SET a.group_norm=a.value/b.sum

The output table name could probably be improved with better logic. After the fact, I changed it from feat$agg_1to3gram$msgsPA_2012$cnty to feat$1to3gram$msgsPA_2012$cnty$10to16$0_01.

