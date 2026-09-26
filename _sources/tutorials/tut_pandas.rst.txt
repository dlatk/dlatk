.. _tut_pandas:
========================
DLATK's Pandas Interface
========================

Importing a FeatureGetter or OutcomeGetter
------------------------------------------
The same methods work for both *FeatureGetter* and *OutcomeGetter*.

.. code-block:: python

	from dlatk.featureGetter import FeatureGetter

	fg = FeatureGetter()  # use defaults set in dlaConstants.py
	fg = FeatureGetter(corpdb="someDB", corptable="someTB", correl_field="someField", ...) # specify values
	fg = FeatureGetter.fromFile('/path/to/init/file') # pass values from file

Init file must have the line `[constants]` at the top. Also note that none of the strings are quoted. For lists (such as lists of outcome variables) use commas to separate values. Sample init file:

.. code-block:: bash

	[constants]
	corpdb = dla_tutorial
	corptable = msgs
	correl_field = user_id
	feattable = feat$1gram$msgs$user_id$16to16$0_01


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
	group_id                         feat                 
	003ae43fae340174a67ffbcf19da1549 neighbors     0.00026
	                                 all           0.00390
	                                 jason         0.00026
	                                 <newline>     0.00130
	                                 caused        0.00026

Get values:

.. code-block:: python

	fg_vals = fg.getValuesAsDF()
	fg_vals.head()
	                                            value
	group_id                         feat            
	003ae43fae340174a67ffbcf19da1549 neighbors      1
	                                 all           15
	                                 jason          1
	                                 <newline>      5
	                                 caused         1

Get group norms with zeros:

.. code-block:: python

	fg_zgns = fg.getGroupNormsWithZerosAsDF()
	fg_zgns.head()
	                                       group_norm
	group_id                         feat            
	003ae43fae340174a67ffbcf19da1549 !       0.096464
	                                 "       0.000780
	                                 #       0.000000
	                                 #12     0.000000
	                                 $       0.000000
	                                 %       0.000000

Create a pivot table:

.. code-block:: python

	fg_zgns_piv = fg.getGroupNormsWithZerosAsDF(pivot=True)
	fg_zgns_piv.head()
	                                       group_norm                                              
	feat                                ¿    –    —    ‘         ’    “    ”    •   
	group_id                                                                        
	003ae43fae340174a67ffbcf19da1549  0.0  0.0  0.0  0.0  0.000000  0.0  0.0  0.0   
	01f6c25f87600f619e05767bf8942a5f  0.0  0.0  0.0  0.0  0.000677  0.0  0.0  0.0   
	02be98c1005c0e7605385fbc5009de61  0.0  0.0  0.0  0.0  0.000000  0.0  0.0  0.0   
	0318cc38971845f7470f34704de7339d  0.0  0.0  0.0  0.0  0.001647  0.0  0.0  0.0   
	040b2b154e4074a72d8a7b9697ec76d2  0.0  0.0  0.0  0.0  0.000000  0.0  0.0  0.0

Create a sparse dataframe:

.. code-block:: python

   fg_sparse = fg.getGroupNormsWithZerosAsDF(sparse=True)
   fg_sparse.density
   0.07432567922874671
 
   fg_sparse.head()
	                                       group_norm
	group_id                         feat            
	003ae43fae340174a67ffbcf19da1549 !       0.096464
	                                 "       0.000780
	                                 #       0.000000
	                                 #12     0.000000
	                                 $       0.000000
	                                 %       0.000000

Outcomes
--------
Init file:

.. code-block:: bash

	[constants]
	corpdb = dla_tutorial
	corptable = msgs
	correl_field = user_id
	feattable = feat$1gram$msgs$user_id$16to16$0_01
	outcometable = blog_outcomes
	outcomefields = age, is_education
	outcomecontrols = gender

Initialize:

.. code-block:: python

	from dlatk.outcomeGetter import OutcomeGetter
	og = OutcomeGetter.fromFile('testInitFile.txt')

Get outcomes and controls:

.. code-block:: python

	outAndCont = og.getGroupsAndOutcomesAsDF()
	outAndCont.head()

	          age  is_education  gender
	group_id                           
	28451      27           NaN       0
	174357     23           NaN       1
	216833     24           NaN       0
	317581     26           NaN       0
	446275     17           NaN       1

	outcome = og.getGroupAndOutcomeValuesAsDF()
	outcome.head()

	         age
	user_id     
	3991108   17
	3417138   25
	3673414   14
	3361075   16
	4115327   14

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

