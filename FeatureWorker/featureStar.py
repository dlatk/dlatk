from ConfigParser import SafeConfigParser
import pandas as pd

import fwConstants as fwc
from featureWorker import FeatureWorker
from featureGetter import FeatureGetter
from featureExtractor import FeatureExtractor
from featureRefiner import FeatureRefiner
from outcomeGetter import OutcomeGetter
from outcomeAnalyzer import OutcomeAnalyzer

class FeatureStar(object):
	"""Generic class for importing all classes in Feature Worker"""

	@classmethod
	def fromFile(cls, initFile, initList=None):
		"""load variables from file"""
		parser = SafeConfigParser()
		parser.read(initFile)
		corpdb = parser.get('constants','corpdb') if parser.has_option('constants','corpdb') else fwc.DEF_CORPDB
		corptable = parser.get('constants','corptable') if parser.has_option('constants','corptable') else fwc.DEF_CORPTABLE
		correl_field = parser.get('constants','correl_field') if parser.has_option('constants','correl_field') else fwc.DEF_CORREL_FIELD
		mysql_host = parser.get('constants','mysql_host') if parser.has_option('constants','mysql_host') else "localhost"
		message_field = parser.get('constants','message_field') if parser.has_option('constants','message_field') else fwc.DEF_MESSAGE_FIELD
		messageid_field = parser.get('constants','messageid_field') if parser.has_option('constants','messageid_field') else fwc.DEF_MESSAGEID_FIELD
		lexicondb = parser.get('constants','lexicondb') if parser.has_option('constants','lexicondb') else fwc.DEF_LEXICON_DB
		featureTable = parser.get('constants','featureTable') if parser.has_option('constants','featureTable') else DEF_FEAT_TABLE
		featNames = parser.get('constants','featNames') if parser.has_option('constants','featNames') else fwc.DEF_FEAT_NAMES
		date_field = parser.get('constants','date_field') if parser.has_option('constants','date_field') else fwc.DEF_DATE_FIELD
		outcome_table = parser.get('constants','outcome_table') if parser.has_option('constants','outcome_table') else fwc.DEF_OUTCOME_TABLE
		outcome_value_fields = [o.strip() for o in parser.get('constants','outcome_value_fields').split(",")] if parser.has_option('constants','outcome_value_fields') else [fwc.DEF_OUTCOME_FIELD] # possible list
		outcome_controls = [o.strip() for o in parser.get('constants','outcome_controls').split(",")] if parser.has_option('constants','outcome_controls') else fwc.DEF_OUTCOME_CONTROLS # possible list
		outcome_interaction = [o.strip() for o in parser.get('constants','outcome_interaction').split(",")] if parser.has_option('constants','outcome_interaction') else fwc.DEF_OUTCOME_CONTROLS # possible list
		featureMappingTable = parser.get('constants','featureMappingTable') if parser.has_option('constants','featureMappingTable') else ''
		featureMappingLex = parser.get('constants','featureMappingLex') if parser.has_option('constants','featureMappingLex') else ''
		output_name = parser.get('constants','output_name') if parser.has_option('constants','output_name') else ''
		wordTable = parser.get('constants','wordTable') if parser.has_option('constants','wordTable') else None
		if initList:
			init = initList
		else:
			init = [o.strip() for o in parser.get('constants','init').split(",")] if parser.has_option('constants','init') else ['fw', 'fg', 'fe', 'fr', 'og', 'oa']
		return cls(corpdb=corpdb, corptable=corptable, correl_field=correl_field, mysql_host=mysql_host, message_field=message_field, messageid_field=messageid_field, lexicondb=lexicondb, featureTable=featureTable, featNames=featNames, date_field=date_field, outcome_table=outcome_table, outcome_value_fields=outcome_value_fields, outcome_controls=outcome_controls, outcome_interaction=outcome_interaction, featureMappingTable=featureMappingTable, featureMappingLex=featureMappingLex,  output_name=output_name, wordTable=wordTable, init=init)
	
	def __init__(self, corpdb=fwc.DEF_CORPDB, corptable=fwc.DEF_CORPTABLE, correl_field=fwc.DEF_CORREL_FIELD, mysql_host="localhost", message_field=fwc.DEF_MESSAGE_FIELD, messageid_field=fwc.DEF_MESSAGEID_FIELD, lexicondb=fwc.DEF_LEXICON_DB, featureTable=fwc.DEF_FEAT_TABLE, featNames=fwc.DEF_FEAT_NAMES, date_field=fwc.DEF_DATE_FIELD, outcome_table=fwc.DEF_OUTCOME_TABLE, outcome_value_fields=[fwc.DEF_OUTCOME_FIELD], outcome_controls = fwc.DEF_OUTCOME_CONTROLS, outcome_interaction = fwc.DEF_OUTCOME_CONTROLS, featureMappingTable='', featureMappingLex='',  output_name='', wordTable=None, init=None):
		
		if init:
			self.fw = FeatureWorker(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, lexicondb, date_field, wordTable) if 'fw' in init else None
			self.fg = FeatureGetter(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, lexicondb, featureTable, featNames, wordTable) if 'fg' in init else None
			self.fe = FeatureExtractor(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, lexicondb, wordTable=wordTable) if 'fe' in init else None
			self.fr = FeatureRefiner(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, lexicondb, featureTable, featNames, wordTable) if 'fr' in init else None
			self.og = OutcomeGetter(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, lexicondb, outcome_table, outcome_value_fields, outcome_controls, outcome_interaction, featureMappingTable, featureMappingLex, wordTable) if 'og' in init else None
			self.oa = OutcomeAnalyzer(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, lexicondb, outcome_table, outcome_value_fields, outcome_controls, outcome_interaction, featureMappingTable, featureMappingLex, output_name, wordTable) if 'oa' in init else None
		else: 
			self.fw = FeatureWorker(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, lexicondb, date_field, wordTable)
			self.fg = FeatureGetter(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, lexicondb, featureTable, featNames, wordTable)
			self.fe = FeatureExtractor(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, lexicondb, wordTable=wordTable)
			self.fr = FeatureRefiner(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, lexicondb, featureTable, featNames, wordTable)
			self.og = OutcomeGetter(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, lexicondb, outcome_table, outcome_value_fields, outcome_controls, outcome_interaction, featureMappingTable, featureMappingLex, wordTable)
			self.oa = OutcomeAnalyzer(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, lexicondb, outcome_table, outcome_value_fields, outcome_controls, outcome_interaction, featureMappingTable, featureMappingLex, output_name, wordTable)
		
		self.allFW = {
				"FeatureWorker": self.fw,
				"FeatureGetter": self.fg,
				"FeatureExtractor": self.fe,
				"FeatureRefiner": self.fr,
				"OutcomeGetter": self.og,
				"OutcomeAnalyzer": self.oa
			}

	def combineDFs(self, fg=None, og=None, fillNA=True):
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
