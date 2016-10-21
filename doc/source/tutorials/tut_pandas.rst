.. _tut_pandas:
========================
DLATK's Pandas Interface
========================

Importing a FeatureGetter or OutcomeGetter
------------------------------------------
The same methods work for both *FeatureGetter* and *OutcomeGetter*.

.. code-block:: python

	from dlatk.featureGetter import FeatureGetter

	fg = FeatureGetter()  # use defaults set in fwConstants.py
	fg = FeatureGetter(corpdb="someDB", corptable="someTB", correl_field="someField", ...) # specify values
	fg = FeatureGetter.fromFile('/path/to/init/file') # pass values from file

Init file must have the line `[constants]` at the top. Also note that none of the strings are quoted. For lists (such as lists of outcome variables) use commas to separate values. Sample init file:

.. code-block:: bash

	[constants]
	corpdb = paHealth
	corptable = messages
	correl_field = user_id
	message_field = message
	messageid_field = message_id
	feattable = feat$1to3gram$msgsPA_2011$cntyYM$16to16$0_01


Getting feature tables as dataframes
------------------------------------

.. code-block:: python

	fg = FeatureGetter()

	fg_gns = fg.getGroupNormsAsDF(where='') # group norms as dataframe
	fg_vals = fg.getValuesAsDF(where='')
	fg_zgns = fg.getGroupNormsWithZerosAsDF(groups=[], where='', pivot=False, sparse=False)
	fg_vgns = fg.getValuesAndGroupNormsAsDF(where='')
    

Getting outcome tables as dataframes
------------------------------------
 
.. code-block:: python

	og = OutcomeGetter()

	# outcome table as dataframe
	og_vals = og.getGroupAndOutcomeValuesAsDF(outcomeField = None, where='') 

	# outcome table as dataframe with group freq thresh applied
	og_out = og.getGroupsAndOutcomesAsDF(groupThresh = 0, lexicon_count_table=None, groupsWhere = '', sparse=False) 
 

Examples
--------
In these examples the testInitFile is the same as the sample init file above. 

Features
^^^^^^^^

.. code-block:: python

	from dlatk.featureGetter import FeatureGetter
	fg = FeatureGetter.fromFile("testInitFile.txt")
 
Get group Norms:

.. code-block:: python

	fg_gns = fg.getGroupNormsAsDF() 
	fg_gns.head()
	                        group_norm
	group_id      feat                   
	42055_2011_12 <OOV_1gram>    0.082580
	42057_2011_12 <OOV_1gram>    0.158879
	42051_2011_12 <OOV_1gram>    0.095937
	42053_2011_12 <OOV_1gram>    0.117647
	42059_2011_12 <OOV_1gram>    0.106342

Get values:

.. code-block:: python

	fg_vals = fg.getValuesAsDF()
	fg_vals.head()
	                        value
	group_id      feat              
	42055_2011_12 <OOV_1gram>    822
	42057_2011_12 <OOV_1gram>     17
	42051_2011_12 <OOV_1gram>    784
	42053_2011_12 <OOV_1gram>      2
	42059_2011_12 <OOV_1gram>    275

Get group norms with zeros:

.. code-block:: python

	fg_zgns = fg.getGroupNormsWithZerosAsDF()
	fg_zgns.head()
	                  group_norm
	group_id      feat             
	42001_2011_12 !        0.015954
	           ! !      0.000000
	           ! ! !    0.000000
	           ! !!     0.000000
	           ! !!!    0.000000

Create a pivot table:

.. code-block:: python

	fg_zgns_piv = fg.getGroupNormsWithZerosAsDF(pivot=True)
	fg_zgns_piv.head()
	           group_norm                                              
	feat                   !       ! !     ! ! !      ! !! ! !!!    ! !!!!   
	group_id                                                                 
	42001_2011_12   0.015954  0.000000  0.000000  0.000000     0  0.000000   
	42003_2011_12   0.012275  0.000014  0.000004  0.000006     0  0.000001   
	42005_2011_12   0.008801  0.000000  0.000000  0.000000     0  0.000000   
	42007_2011_12   0.010857  0.000000  0.000000  0.000000     0  0.000000   
	42009_2011_12   0.012678  0.000000  0.000000  0.000000     0  0.000000

Create a sparse dataframe:

.. code-block:: python

   g_sparse = fg.getGroupNormsWithZerosAsDF(sparse=True)
   fg_sparse.density
   0.054158277796754875
 
   fg_sparse.head()
                         group_norm
   group_id      feat             
   42001_2011_12 !        0.015954
                 ! !      0.000000
                 ! ! !    0.000000
                 ! !!     0.000000
                 ! !!!    0.000000

Outcomes
--------
Init file:

.. code-block:: bash

	[constants]
	corpdb = paHealth
	corptable = msgsPA
	correl_field = cnty
	message_field = message
	messageid_field = message_id
	outcometable = outcomes
	outcomefields = ED_perc, AIDD_perc
	outcomecontrols = age_1to4, age_5to9
	feattable = feat$1to3gram$msgsPA_2013$cnty$16to16$0_01

Initialize:

.. code-block:: python

	from dlatk.outcomeGetter import OutcomeGetter
	og = OutcomeGetter.fromFile('testInitFile.txt')

Get outcomes and controls:

.. code-block:: python

	outAndCont = og.getGroupsAndOutcomesAsDF()
	outAndCont.head()

	        AIDD_perc  ED_perc  age_1to4  age_5to9
	42001         41       17  0.044758  0.060233
	42003         27       19  0.041452  0.052760
	42005         30       21  0.042384  0.053213
	42007         35       14  0.042023  0.054353
	42009         31        8  0.042326  0.059348

	outcome = og.getGroupAndOutcomeValuesAsDF()
	outcome.head()

	        ED_perc
	cnty          
	42001       17
	42003       19
	42005       21
	42007       14
	42009        8

Features and Outcomes in one dataframe
--------------------------------------
Initialize:

.. code-block:: python

	from dlatk.featureStar import FeatureStar
	fs = FeatureStar.fromFile('testInitFile.txt')

Get both dataframe with all info:

.. code-block:: python

	fAndO_df = fs.combineDFs(fg=None, og=None, fillNA=True)

**fg** can be either a *FeatureGetter* or a dataframe with index on **group_id**. Similarly, **og** can be either a *OutcomeGetter* or a dataframe with index on **group_id**. Alternatively, you can pass nothing to the method, which will return a dataframe with with data from the feature and outcome tables in *FeatureStar*.

.. code-block:: python

	fAndO  = fs.combineDFs() # pass nothing
	fAndO  = fs.combineDFs(someFeatureGetter, someOutcomeGetter) # pass objects
	fAndO  = fs.combineDFs(someFeatureDF, someOutcomeDF) # pass dataframes

