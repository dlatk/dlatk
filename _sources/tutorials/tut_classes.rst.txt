.. _tut_classes:
============================
Working with DLATK's Classes
============================

DLATK contains the following classes:  *FeatureWorker*, *FeatureGetter*, *FeatureRefiner*, *FeatureExtractor*, *OutcomeGetter*, *OutcomeAnalyzer*, *ClassifyPredictor* and *RegressionPredictor*. Instead of working with dlatkInterface one can work directly with one of these classes. 

Working with classes
--------------------
With the exception of *FeatureWorker* all classes can be created as follows:

.. code-block:: python

	from dlatk.featureExtractor import FeatureExtractor
	fe = FeatureExtractor() # use default values set in dlaConstants
	fe = FeatureExtractor(corpdb="someDB", corptable="someTB", correl_field="someField", ...) # specify values

*FeatureWorker* does not allow you to use default values so you must do the following:

.. code-block:: python

	from dlatk.featureWorker import FeatureWorker
	fw = FeatureWorker(corpdb="someDB", corptable="someTB", correl_field="someField", ...) 

Both  *FeatureGetter*, *OutcomeGetter* and *OutcomeAnalyzer* allow you to pass an init file. This file is a simple text file where the constants are defined. Init file must have the line **[constants]** at the top. Also note that none of the strings are quoted. For lists (such as lists of outcome variables) use commas to separate values. Sample init file:

.. code-block:: bash

	[constants]
	corpdb = dla_tutorial
	corptable = msgs
	correl_field = user_id
	feattable = feat$1gram$msgs$user_id$16to16$0_01
	outcometable = blog_outcomes
	outcomefields = age, is_education
	outcomecontrols = gender

To initialize one of these classes using an init file do the following:

.. code-block:: python

	fg = FeatureGetter.fromFile('/path/to/init/file')
	og = OutcomeGetter.fromFile('/path/to/init/file')
	oa = OutcomeAnalyzer.fromFile('/path/to/init/file')

Using FeatureStar
-----------------
This option allows you to create instances of *FeatureWorker*, *FeatureGetter*, *FeatureRefiner*, *FeatureExtractor*, *OutcomeGetter*, *OutcomeAnalyzer*, *ClassifyPredictor* and *RegressionPredictor*. 

Initialize a FeatureStar object:

.. code-block:: python

	from dlatk.featureStar import FeatureStar

	# get instances of all classes
	fs = FeatureStar() # defaults specified in dlaConstants
	fs = FeatureStar(corpdb="someDB", corptable="someTB", correl_field="someField", ...) # specify values
	fs = FeatureStar.fromFile('/path/to/init/file') # pass values from file

To initalize FeatureStar with an init file, simply add the following line under **[constants]**:

.. code-block:: python

	init = fg, og

This is a comma separated line with the initials of the classes you want to instantiate. So the above command will give you a FeatureGetter and an OutcomeGetter

.. code-block:: python

	# add line 'init = fg, og' to init file
	fs = FeatureStar.fromFile('/path/to/init/file') # this will return only a FeatureGetter and OutcomeGetter

Or you can pass this list as a parameter

.. code-block:: python

	# get certain classes with defaults
	fs = FeatureStar(init=['fg', 'og'])

	# get instances of subset of classes
	fs = FeatureStar.fromFile('/path/to/init/file', ['fg', 'og']) # get only a FeatureGetter and OutcomeGetter

	fs.allFW
	> {'FeatureWorker': None, 'FeatureRefiner': None, 
	'FeatureGetter': <FeatureWorker.featureGetter.FeatureGetter object at 0x7f87e75e13d0>, 
	'OutcomeGetter': <FeatureWorker.outcomeGetter.OutcomeGetter object at 0x7f87e75e1390>, 
	'OutcomeAnalyzer': None, 'FeatureExtractor': None,
	'ClassifyPredictor': None, 'RegressionPredictor': None}


You can retrieve FeatureStars attributes (i.e., all DLATK objects) in the following way

.. code-block:: python

	# retrieve a FeatureWorker
	fw = fs.fw
	fw = fs.allFW['FeatureWorker']

	# retrieve a FeatureGetter
	fg = fs.fg
	fg = fs.allFW['FeatureGetter']

	# retrieve a FeatureRefiner
	fr = fs.fr
	fr = fs.allFW['FeatureRefiner']

	# retrieve a FeatureExtractor
	fe = fs.fe
	fe = fs.allFW['FeatureExtractor']

	# retrieve a OutcomeGetter
	og = fs.og
	og = fs.allFW['OutcomeGetter']

	# retrieve a OutcomeAnalyzer
	oa = fs.oa
	oa = fs.allFW['OutcomeAnalyzer']

	# retrieve a RegressionPredictor
	oa = fs.rp
	oa = fs.allFW['RegressionPredictor']

	# retrieve a ClassifyPredictor
	oa = fs.cp
	oa = fs.allFW['ClassifyPredictor']

