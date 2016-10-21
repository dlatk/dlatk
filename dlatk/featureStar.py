import pandas as pd
from configparser import SafeConfigParser

#infrastructure
from . import fwConstants as fwc
from .featureWorker import FeatureWorker
from .featureGetter import FeatureGetter
from .featureExtractor import FeatureExtractor
from .featureRefiner import FeatureRefiner
from .outcomeGetter import OutcomeGetter
from .outcomeAnalyzer import OutcomeAnalyzer
from .regressionPredictor import RegressionPredictor
from .classifyPredictor import ClassifyPredictor

class FeatureStar(object):
	"""Generic class for importing an instance of each class in Feature Worker

    Parameters
    ----------
    fw : FeatureWorker object
    
	fe : FeatureExtractor object

	fg : FeatureGetter object

	fr : FeatureRefiner object

	og : OutcomeGetter object

	oa : OutcomeAnalyzer object

	cp : ClassifyPredictor object

	rp : RegressionPredictor object

	allFW : dict
		Dictionary containing all of the above attributes keyed on object name

    Examples
    --------
    Initialize a FeatureStar

    >>> fs = FeatureStar.fromFile('~/myInit.ini')

    Create a pandas dataframe with both feature and outcome information

    >>> df = fs.combineDFs()

    """


	@classmethod
	def fromFile(cls, initFile, initList=None):
		"""Loads specified features from file

        Parameters
        ----------
        initFile : str 
            Path to file

        initList : list
        	List of classes to load

        """
		parser = SafeConfigParser()
		parser.read(initFile)
		corpdb = parser.get('constants','corpdb') if parser.has_option('constants','corpdb') else fwc.DEF_CORPDB
		corptable = parser.get('constants','corptable') if parser.has_option('constants','corptable') else fwc.DEF_CORPTABLE
		correl_field = parser.get('constants','correl_field') if parser.has_option('constants','correl_field') else fwc.DEF_CORREL_FIELD
		print(correl_field)
		mysql_host = parser.get('constants','mysql_host') if parser.has_option('constants','mysql_host') else fwc.MYSQL_HOST
		message_field = parser.get('constants','message_field') if parser.has_option('constants','message_field') else fwc.DEF_MESSAGE_FIELD
		messageid_field = parser.get('constants','messageid_field') if parser.has_option('constants','messageid_field') else fwc.DEF_MESSAGEID_FIELD
		encoding = parser.get('constants','encoding') if parser.has_option('constants','encoding') else fwc.DEF_ENCODING
		if parser.has_option('constants','use_unicode'):
			use_unicode = True if parser.get('constants','use_unicode')=="True" else False
		else:
			use_unicode = fwc.DEF_UNICODE_SWITCH
		lexicondb = parser.get('constants','lexicondb') if parser.has_option('constants','lexicondb') else fwc.DEF_LEXICON_DB
		if parser.has_option('constants','feattable'):
			if len(parser.get('constants','feattable').split(",")) > 0:
				featureTable = [f.strip() for f in parser.get('constants','feattable').split(",")]
			else:
				featureTable = parser.get('constants','feattable')
		else:
			featureTable = fwc.DEF_FEAT_TABLE
		print(featureTable)
		print(fwc.DEF_FEAT_TABLE)
		featNames = parser.get('constants','featnames') if parser.has_option('constants','featnames') else fwc.DEF_FEAT_NAMES
		date_field = parser.get('constants','date_field') if parser.has_option('constants','date_field') else fwc.DEF_DATE_FIELD
		outcome_table = parser.get('constants','outcometable') if parser.has_option('constants','outcometable') else fwc.DEF_OUTCOME_TABLE
		print(outcome_table)
		outcome_value_fields = [o.strip() for o in parser.get('constants','outcomefields').split(",")] if parser.has_option('constants','outcomefields') else [fwc.DEF_OUTCOME_FIELD] # possible list
		outcome_controls = [o.strip() for o in parser.get('constants','outcomecontrols').split(",")] if parser.has_option('constants','outcomecontrols') else fwc.DEF_OUTCOME_CONTROLS # possible list
		print(outcome_controls)
		outcome_interaction = [o.strip() for o in parser.get('constants','outcomeinteraction').split(",")] if parser.has_option('constants','outcomeinteraction') else fwc.DEF_OUTCOME_CONTROLS # possible list
		group_freq_thresh = int(parser.get('constants','groupfreqthresh')) if parser.has_option('constants','groupfreqthresh') else fwc.getGroupFreqThresh(correl_field)
		featureMappingTable = parser.get('constants','featlabelmaptable') if parser.has_option('constants','featlabelmaptable') else ''
		featureMappingLex = parser.get('constants','featlabelmaplex') if parser.has_option('constants','featlabelmaplex') else ''
		output_name = parser.get('constants','outputname') if parser.has_option('constants','outputname') else ''
		wordTable = parser.get('constants','wordTable') if parser.has_option('constants','wordTable') else None
		model =  parser.get('constants','model') if parser.has_option('constants','model') else fwc.DEF_MODEL
		feature_selection = fwc.DEF_FEATURE_SELECTION_MAPPING[parser.get('constants','featureselection')] if parser.has_option('constants','featureselection') else ''
		feature_selection_string = parser.get('constants','featureselectionstring') if parser.has_option('constants','featureselectionstring') else ''
		if initList:
			init = initList
		else:
			init = [o.strip() for o in parser.get('constants','init').split(",")] if parser.has_option('constants','init') else ['fw', 'fg', 'fe', 'fr', 'og', 'oa', 'rp', 'cp']
		return cls(corpdb=corpdb, corptable=corptable, correl_field=correl_field, mysql_host=mysql_host, message_field=message_field, messageid_field=messageid_field, encoding=encoding, use_unicode=use_unicode, lexicondb=lexicondb, featureTable=featureTable, featNames=featNames, date_field=date_field, outcome_table=outcome_table, outcome_value_fields=outcome_value_fields, outcome_controls=outcome_controls, outcome_interaction=outcome_interaction, group_freq_thresh=group_freq_thresh, featureMappingTable=featureMappingTable, featureMappingLex=featureMappingLex,  output_name=output_name, wordTable=wordTable, model=model, feature_selection=feature_selection, feature_selection_string = feature_selection_string, init=init)
	
	def __init__(self, corpdb=fwc.DEF_CORPDB, corptable=fwc.DEF_CORPTABLE, correl_field=fwc.DEF_CORREL_FIELD, mysql_host=fwc.MYSQL_HOST, message_field=fwc.DEF_MESSAGE_FIELD, messageid_field=fwc.DEF_MESSAGEID_FIELD, encoding=fwc.DEF_ENCODING, use_unicode=fwc.DEF_UNICODE_SWITCH, lexicondb=fwc.DEF_LEXICON_DB, featureTable=fwc.DEF_FEAT_TABLE, featNames=fwc.DEF_FEAT_NAMES, date_field=fwc.DEF_DATE_FIELD, outcome_table=fwc.DEF_OUTCOME_TABLE, outcome_value_fields=[fwc.DEF_OUTCOME_FIELD], outcome_controls = fwc.DEF_OUTCOME_CONTROLS, outcome_interaction = fwc.DEF_OUTCOME_CONTROLS, group_freq_thresh = None, featureMappingTable='', featureMappingLex='',  output_name='', wordTable=None, model=fwc.DEF_MODEL, feature_selection='', feature_selection_string = '', init=None):
		
		if feature_selection_string or feature_selection:
			RegressionPredictor.featureSelectionString = feature_selection if feature_selection else feature_selection_string

		if init:
			self.fw = FeatureWorker(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, date_field, wordTable) if 'fw' in init else None
			if 'fg' in init:
				if isinstance(featureTable, str):
					self.fg = FeatureGetter(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, featureTable, featNames, wordTable)
				else:
					self.fg = [FeatureGetter(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, ft, featNames, wordTable) for ft in featureTable]
			else:
				None
			self.fe = FeatureExtractor(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, wordTable=wordTable) if 'fe' in init else None
			self.fr = FeatureRefiner(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, featureTable, featNames, wordTable) if 'fr' in init else None
			self.og = OutcomeGetter(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, outcome_table, outcome_value_fields, outcome_controls, outcome_interaction, group_freq_thresh, featureMappingTable, featureMappingLex, wordTable) if 'og' in init else None
			self.oa = OutcomeAnalyzer(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, outcome_table, outcome_value_fields, outcome_controls, outcome_interaction, group_freq_thresh, featureMappingTable, featureMappingLex, output_name, wordTable) if 'oa' in init else None
			self.rp = RegressionPredictor(self.og, self.fg, model) if 'rp' in init else None
			self.cp = ClassifyPredictor(self.og, self.fg, model) if 'cp' in init else None

		else: 
			self.fw = FeatureWorker(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, date_field, wordTable)
			if isinstance(featureTable, str):
				self.fg = FeatureGetter(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, featureTable, featNames, wordTable)
			else:
				self.fg = [FeatureGetter(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, ft, featNames, wordTable) for ft in featureTable]
			self.fe = FeatureExtractor(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, wordTable=wordTable)
			self.fr = FeatureRefiner(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, featureTable, featNames, wordTable)
			self.og = OutcomeGetter(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, outcome_table, outcome_value_fields, outcome_controls, outcome_interaction, group_freq_thresh, featureMappingTable, featureMappingLex, wordTable)
			self.oa = OutcomeAnalyzer(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, outcome_table, outcome_value_fields, outcome_controls, outcome_interaction, group_freq_thresh, featureMappingTable, featureMappingLex, output_name, wordTable)
			self.rp = RegressionPredictor(self.og, self.fg, model)
			self.cp = ClassifyPredictor(self.og, self.fg, model)
		
		self.allFW = {
				"FeatureWorker": self.fw,
				"FeatureGetter": self.fg,
				"FeatureExtractor": self.fe,
				"FeatureRefiner": self.fr,
				"OutcomeGetter": self.og,
				"OutcomeAnalyzer": self.oa,
				"RegressionPredictor": self.rp,
				"ClassifyPredictor": self.cp,
			}

	def combineDFs(self, fg=None, og=None, fillNA=True):
		"""Method for combining a feature table with an outcome table in a single dataframe

        Parameters
        ----------
        fg : FeatureGetter object
        
        og : OutcomeGetter object
        
        fillNA : boolean)
			option to fill missing or NA values in dataframe, fill value = 0

        Returns
        -------
        pandas dataframe 
        	Dataframe indexed on group_id (correl_field)

        """
		if fg:
			if isinstance(fg, FeatureGetter):
				fg = fg.getGroupNormsWithZerosAsDF(pivot=True)
		else:
			fg = self.fg.getGroupNormsWithZerosAsDF(pivot=True)
		if og:
			if isinstance(og, OutcomeGetter):
				og = og.getGroupsAndOutcomesAsDF()
		else:
			og = self.og.getGroupsAndOutcomesAsDF()

		if fillNA:
			return pd.concat([fg, og], axis=1).fillna(value=0)
		else:
			return pd.concat([fg, og], axis=1)
