.. _tut_feat_tables:
=================================
Understanding Feature Table Names
=================================

This page will explain the **standard** way of naming feature tables in DLATK.

This is how DLATK expects them to be named.

Deviate at your own risk.

Structure
---------

Every feature table has the same structure: `id`, `group_id`, `feat`, `value` and `group_norm`. Here is an example of a message level (`group_id` = `message_id`) 1gram table:

.. code-block:: mysql 

   	mysql> describe feat$1gram$msgs$message_id;
	+------------+---------------------+------+-----+---------+----------------+
	| Field      | Type                | Null | Key | Default | Extra          |
	+------------+---------------------+------+-----+---------+----------------+
	| id         | bigint(16) unsigned | NO   | PRI | NULL    | auto_increment |
	| group_id   | int(11)             | YES  | MUL | NULL    |                |
	| feat       | varchar(36)         | YES  | MUL | NULL    |                |
	| value      | int(11)             | YES  |     | NULL    |                |
	| group_norm | double              | YES  |     | NULL    |                |
	+------------+---------------------+------+-----+---------+----------------+

The column naming convention is identical across tables but the MySQL Type is not, though generally `feat` is a `varchar`, `value` is an `int` and `group_norm` is a `double`. The columns are defined as follows:

* **group_id**: Identifier for each group as determined from the :doc:`../fwinterface/fwflag_c` flag. This is typically a message id (e.g. Tweet id), user id (e.g. Twitter user id), community id (e.g. U.S. County FIPS code or state code), etc.
* **feat**: feature name such as an ngram, LDA topic id, etc.
* **value**: The number of times the feature was used by the `group_id`.
* **group_norm**: The relative frequency of the feature use for the `group_id`. This is usually `value` divided by the sum of all `value`s for the `group_id`.

Things to keep in mind when creating your own feature tables:

* The `id` column is technically not necessary but every other column is needed. 
* Tables are sparse encoded: `group_id` / `feat` pairs are assumed to be zero if missing from the table.
* Nulls and 0's in the `group_norm` column will throw an error.
* Do not use `Decimal` types in feature tables.
* Keep the `group_id` and `feat` columns indexed.

Example 1: unigram, bigram, etc features
----------------------------------------
These tables are generally created with the :doc:`../fwinterface/fwflag_add_ngrams` flag of fwInterface.

.. code-block:: bash

	feat$1to3gram$statuses_er1$user_id$16to1$0_01$pmi3_0
	| f0 |field 1 |  field 2   |field3 |field4| f5 |field 6|

**Field 0** Specifies this as a feature table. All feature tables begin with the word "feat".

**Field 1** Specifies kinds of features; these are 1-, 2-, and 3-grams, the result of running :doc:`../fwinterface/fwflag_combine_feat_tables` after :doc:`../fwinterface/fwflag_add_ngrams`

**Field 2** Gives the message table (:doc:`../fwinterface/fwflag_t`) that the features were derived from

**Field 3** Gives the group ID (:doc:`../fwinterface/fwflag_c`) that features were grouped by

**Field 4** Specifies scaling on features. The default (or unscaled) feature tables do not include this field.

* *16to8*: :doc:`../fwinterface/fwflag_anscombe`
* *16to4*: :doc:`../fwinterface/fwflag_sqrt`
* *16to3*: :doc:`../fwinterface/fwflag_log`
* *16to1*: :doc:`../fwinterface/fwflag_boolean`

**Field 5** Shows feature occurrence filter (:doc:`../fwinterface/fwflag_feat_occ_filter`) used on feature table (i.e., what %age of groups necessary to include feature in table)

**Field 6** Gives the PMI threshold set by :doc:`../fwinterface/fwflag_feat_colloc_filter`, and optionally, :doc:`../fwinterface/fwflag_set_pmi_threshold`

Example 2: extracted lexicon/topic features
-------------------------------------------
These tables are generally created with the :doc:`../fwinterface/fwflag_add_lex_table` flag of dlatkInterface.

.. code-block:: bash

	feat$cat_met_a30_2000_cp_w$messages_en$cty_id$1gra
	| f0 |       field 1      |  field 2  |field3|field4|

**Field 0** Specifies this as a feature table

**Field 1** Specifies the source of features; these are extracted from the topic lexicon *met_a30_2000*, and the table was created via :doc:`../fwinterface/fwflag_add_lex_table`. The trailing "*_w*" indicates a weighted lexicon. "*_cp*" stands for "conditional probability", one of the two types of topic lexica normally created (see :doc:`../tutorials/tut_lda`).

**Field 2** Gives the message table (:doc:`../fwinterface/fwflag_t`) that the features were derived from

**Field 3** Gives the group ID (:doc:`../fwinterface/fwflag_c`) that features were grouped by

**Field 4** The first four characters from Field 1 of the word table (:doc:`../fwinterface/fwflag_word_table`)  used to derive the lexicon/topic features. By default this is the 1gram table. In previous version (less than 1.1.5) this field specified the scaling on features.
