.. _tut_feat_tables:
=================================
Understanding Feature Table Names
=================================

This page will explain the **standard** way of naming feature tables in DLATK.

This is how DLATK expects them to be named.

Deviate at your own risk.

Example 1: unigram, bigram, etc features
----------------------------------------
These tables are generally created with the :doc:`../fwinterface/fwflag_add_ngrams` flag of fwInterface.

.. code-block:: bash

	feat$1to3gram$statuses_er1$user_id$16to16$0_01$pmi3_0
	| f0 |field 1 |  field 2   |field3 |field4| f5 |field 6|

**Field 0** Specifies this as a feature table

**Field 1** Specifies kinds of features; these are 1-, 2-, and 3-grams, the result of running :doc:`../fwinterface/fwflag_combine_feat_tables` after :doc:`../fwinterface/fwflag_add_ngrams`

**Field 2** Gives the message table (:doc:`../fwinterface/fwflag_t`) that the features were derived from

**Field 3** Gives the group ID (:doc:`../fwinterface/fwflag_c`) that features were grouped by

**Field 4** Specifies scaling on features:

* *16to16*: Unscaled
* *16to8*: :doc:`../fwinterface/fwflag_anscombe`
* *16to4*: :doc:`../fwinterface/fwflag_sqrt`
* *16to2*: :doc:`../fwinterface/fwflag_log`
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
