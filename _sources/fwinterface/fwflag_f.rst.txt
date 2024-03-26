.. _fwflag_f:
==
-f
==
Switch
======

-f, --feat_table

Description
===========

Specify where the features are stored

Argument and Default Value
==========================

Name of the feature table in MySQL. For more information, see "Understanding Feature Table Names".

Details
=======

Specifies the name of the table containing extracted features. Features are usually created using one of the steps in Feature Extraction.

Expected column format of the table must/will contain at least the following:
	
	* group_id: Contains the "group_id" id-s (i.e. type specified by :doc:`fwflag_c`). Is an index.
	* feat: Contains the feature type. Could be n-grams, lexicon features, etc.
	* value: Contains the straight up count of the feature for the group_id. For ngrams, is just occurrence count. For lexical features, is number of words matched for given feature & group_id.
	* group_norm: Contain the "relative frequency" of each feature. See specific feature extraction pages for how it actually works.