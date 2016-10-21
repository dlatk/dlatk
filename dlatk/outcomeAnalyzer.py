import os
import sys
import re
import csv
import gzip
import multiprocessing
from itertools import combinations
from pprint import pprint
from configparser import SafeConfigParser 

#math / stats:
from math import floor
from numpy import array, tile, sqrt, fabs, multiply, mean, isnan
from numpy import log as nplog, sort as npsort, append as npappend
import numpy as np
from scipy.stats import zscore, rankdata, norm
from scipy.stats.stats import pearsonr, spearmanr
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

#infrastructure
from .outcomeGetter import OutcomeGetter
from . import fwConstants as fwc 
from .mysqlMethods import mysqlMethods as mm


class OutcomeAnalyzer(OutcomeGetter):
    """
    Deals with Outcome Tables and provides various
    functionalities for how outcomes are processed/analyzed.

    Attributes
    ----------
    output_name: str


    Returns
    -------
    OutcomeAnalyzer object.
    """

    # correl indices
    r_idx = 0
    p_idx = 1 
    n_idx = 2
    ci_idx = 3
    freq_idx = 4
    correls_length = 5


    @classmethod
    def fromFile(cls, initFile):
        """
        Load specified features from INI file


        Parameters
        ----------
        initFile: str
            path to file


        Example
        -------
        creates outcomeAnalyzer Object with features
        specified in the initFile

        >>> outcomeAnalyzer.fromFile('~/myInit.ini')

        """
        parser = SafeConfigParser()
        parser.read(initFile)
        corpdb = parser.get('constants','corpdb') if parser.has_option('constants','corpdb') else fwc.DEF_CORPDB
        corptable = parser.get('constants','corptable') if parser.has_option('constants','corptable') else fwc.DEF_CORPTABLE
        correl_field = parser.get('constants','correl_field') if parser.has_option('constants','correl_field') else fwc.DEF_CORREL_FIELD
        mysql_host = parser.get('constants','mysql_host') if parser.has_option('constants','mysql_host') else fwc.MYSQL_HOST
        message_field = parser.get('constants','message_field') if parser.has_option('constants','message_field') else fwc.DEF_MESSAGE_FIELD
        messageid_field = parser.get('constants','messageid_field') if parser.has_option('constants','messageid_field') else fwc.DEF_MESSAGEID_FIELD
        encoding = parser.get('constants','encoding') if parser.has_option('constants','encoding') else fwc.DEF_ENCODING
        use_unicode = True if parser.get('constants','use_unicode')=="True" else False if parser.has_option('constants','use_unicode') else fwc.DEF_UNICODE_SWITCH
        lexicondb = parser.get('constants','lexicondb') if parser.has_option('constants','lexicondb') else fwc.DEF_LEXICON_DB
        outcome_table = parser.get('constants','outcometable') if parser.has_option('constants','outcometable') else fwc.DEF_OUTCOME_TABLE
        outcome_value_fields = [o.strip() for o in parser.get('constants','outcomefields').split(",")] if parser.has_option('constants','outcomefields') else [fwc.DEF_OUTCOME_FIELD] # possible list
        outcome_controls = [o.strip() for o in parser.get('constants','outcomecontrols').split(",")] if parser.has_option('constants','outcomecontrols') else fwc.DEF_OUTCOME_CONTROLS # possible list
        outcome_interaction = [o.strip() for o in parser.get('constants','outcome_interaction').split(",")] if parser.has_option('constants','outcome_interaction') else fwc.DEF_OUTCOME_CONTROLS # possible list
        group_freq_thresh = parser.get('constants','groupfreqthresh') if parser.has_option('constants','groupfreqthresh') else fwc.getGroupFreqThresh(correl_field)
        featureMappingTable = parser.get('constants','featlabelmaptable') if parser.has_option('constants','featlabelmaptable') else ''
        featureMappingLex = parser.get('constants','featlabelmaplex') if parser.has_option('constants','featlabelmaplex') else ''
        output_name = parser.get('constants','output_name') if parser.has_option('constants','output_name') else ''
        wordTable = parser.get('constants','wordTable') if parser.has_option('constants','wordTable') else None
        return cls(corpdb=corpdb, corptable=corptable, correl_field=correl_field, mysql_host=mysql_host, message_field=message_field, messageid_field=messageid_field, encoding=encoding, use_unicode=use_unicode, lexicondb=lexicondb, outcome_table=outcome_table, outcome_value_fields=outcome_value_fields, outcome_controls=outcome_controls, outcome_interaction=outcome_interaction, group_freq_thresh = group_freq_thresh, featureMappingTable=featureMappingTable, featureMappingLex=featureMappingLex,  output_name=output_name, wordTable=wordTable)
    
    def __init__(self, corpdb=fwc.DEF_CORPDB, corptable=fwc.DEF_CORPTABLE, correl_field=fwc.DEF_CORREL_FIELD, mysql_host=fwc.MYSQL_HOST, message_field=fwc.DEF_MESSAGE_FIELD, messageid_field=fwc.DEF_MESSAGEID_FIELD, encoding=fwc.DEF_ENCODING, use_unicode=fwc.DEF_UNICODE_SWITCH, lexicondb=fwc.DEF_LEXICON_DB, outcome_table=fwc.DEF_OUTCOME_TABLE, outcome_value_fields=[fwc.DEF_OUTCOME_FIELD], outcome_controls=fwc.DEF_OUTCOME_CONTROLS, outcome_interaction=fwc.DEF_OUTCOME_CONTROLS, group_freq_thresh = None, featureMappingTable='', featureMappingLex='',  output_name='', wordTable = None):
        super(OutcomeAnalyzer, self).__init__(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, outcome_table, outcome_value_fields, outcome_controls, outcome_interaction, group_freq_thresh, featureMappingTable, featureMappingLex,  wordTable)
        self.output_name = output_name

    def printGroupsAndOutcomesToCSV(self, featGetter, outputfile, where = '', freqs = False):
        """
        Prints sas-style csv file output

        Parameters
        ----------
            featGetter : featureGetter object
            outputfile : str
                File name for the outcome to be printed to
            where : :obj:`str`, optional
                Additional options based on a `where` statement, similar to mysql select * from table where ....
            freqs : :obj:`boolean`, optional
                if true feature values are returned, if false group norms


        Returns
        ------
        No return value - Writes to a CSV file containing groups and outcomes


        Example
        --------
        Print groups and outcomes to csv
        where fg is a feature getter object and args.printcsv
        is a user defined filename supplied to fwInterface.

        >>> oa.printGroupsAndOutcomesToCSV(fg, args.printcsv)

        """
        assert mm.tableExists(self.corpdb, self.dbCursor, featGetter.featureTable, charset=self.encoding, use_unicode=self.use_unicode), 'feature table does not exist (make sure to quote it)'

        #get outcome data to work with
        (groups, allOutcomes, controls) = OutcomeGetter.getGroupsAndOutcomes()
        print("LENGTH OF GROUPS!! %d" % len(groups))
        allFeats = featGetter.getDistinctFeatures(where)

        #adjust keys for outcomes and controls:
        allOutcomes = dict([('outcome_'+key, value) for key, value in allOutcomes.items()])
        controls = dict([('cntrl_'+key, value) for key, value in controls.items()])

        #create all keys
        allKeys = list(['group_id'])#row label header
        allKeys.extend(list(allOutcomes.keys()))#outcome header
        allKeys.extend(list(controls.keys())) #controls header
        allKeys.extend(allFeats) #features header
    
        #write csv:
        csvOut = csv.DictWriter(open(outputfile, 'w'), fieldnames=allKeys)
        outcomesByGroup = dict()
        if allOutcomes and len(allOutcomes) > 0: outcomesByGroup = fwc.reverseDictDict(allOutcomes)
        controlsByGroup = dict()
        if controls and len(controls) > 0: controlsByGroup = fwc.reverseDictDict(controls)
        firstRow = dict([(k, k) for k in allKeys])
        csvOut.writerow(firstRow)
        numPed = 0
        #can also use yieldGroupNorms if preferring to output that information
        yielder = []
        if freqs:
            yielder = featGetter.yieldValuesWithZerosByGroup(groups, allFeats = allFeats)
        else:
            yielder = featGetter.yieldGroupNormsWithZerosByGroup(groups, allFeats = allFeats)
        for (group, featGns) in yielder:
            rowDict = featGns #.copy
            if group in outcomesByGroup:
                rowDict.update(outcomesByGroup[group])
            if group in controlsByGroup:
                rowDict.update(controlsByGroup[group])
            rowDict['group_id'] = str(group)
            csvOut.writerow(rowDict)
            numPed += 1
            if numPed % 1000 == 0: fwc.warn("  %d groups printed"%(numPed))

    def printBinnedGroupsAndOutcomesToCSV(self, featGetter, outputfile, where = '', freqs = False):
        """
        Print csv file  with binned groups and output

        Parameters
        ----------
            featGetter : featureGetter object
            outputfile : str
                File name for the outcome to be printed to
            where : :obj:`str`, optional
                Additional options based on a `where` statement, similar to mysql select * from table where ....
            freqs : :obj:`boolean`, optional
                if true feature values are returned, if false group norms


        Returns
        ------
        No return value - Raises an Error


        """
        raise NotImplementedError()


    def yieldDataForOneFeatAtATime(self, featGetter, blacklist=None, whitelist=None, outcomeWithOutcome=False, includeFreqs = False, groupsWhere = '', outcomeWithOutcomeOnly = False, ):
        """
        Finds the correlations between features and outcomes

        Parameters
        ----------
            featGetter : featureGetter object
            blacklist : :obj:`list`, optional
                list of feature table fields (str) to ignore
            whitelist : :obj:`list`, optional
                list of feature table fields (str) to include
            outcomeWithOutcome : :obj:`boolean`, optional
                Adds the outcomes themselves to the list of variables to correlate with the outcomes if True
            includeFreqs : :obj:`boolean`, optional
                Include the frequency of each feature if True
            groupsWhere : :obj:`str`, optional
                string specified with the --where flag containing 
            outcomeWithOutcomeOnly : :obj:`boolean`, optional
                True - only correlate outcomes with outcomes
                False - correlate features with outcomes


        Yields
        ------
        groups : 
            contains all groups looked at -> ie all users
        allOutcomes : 
            contains a dictionary of the outcomes and their values for each group in groups
        dataDict : 
            contains the group_norms (i.e what we're z-scoring) for every feature
        controls : 
            dict of controls and counts
        numOutcomes :
        featFreqs :

        """
        if not outcomeWithOutcomeOnly:
            assert mm.tableExists(self.corpdb, self.dbCursor, featGetter.featureTable, charset=self.encoding, use_unicode=self.use_unicode), 'feature table does not exist (make sure to quote it)'
        lexicon_count_table = None
        # if 'cat_' in featGetter.featureTable.split('$')[1]:
        #     lexicon_count_table = featGetter.featureTable
        # LAD TODO NOTE: should implement the whitelist above so it doesn't take so long to select the groups and outcomes...

        #get outcome data to work with
        (groups, allOutcomes, controls) = self.getGroupsAndOutcomes(lexicon_count_table, groupsWhere)
        
        assert len(groups) > 0, "Something is wrong, there aren't any groups left. Maybe the group_freq_thresh is too high, maybe your group field columns are different types"
        featFreqs = None
        if includeFreqs:
            where = """ group_id in ('%s')""" % ("','".join(str(g) for g in groups))
            if outcomeWithOutcomeOnly:
                featFreqs = dict([('outcome_'+k, len(v)) for k, v in allOutcomes.items()])
            else:
                featFreqs = dict([ (k, v) for k, v in  featGetter.getSumValuesByFeat(where = where) ])
                if outcomeWithOutcome :
                    featFreqs.update(dict([('outcome_'+k, len(v)) for k, v in allOutcomes.items()]))
                
        #run correlations:
        fwc.warn("Yielding data to correlate over %s, adjusting for: %s%s" % (str(self.outcome_value_fields),
                                                                            str(self.outcome_controls),
                                                                            " interaction with "+str(self.outcome_interaction) if self.outcome_interaction else "."))
        if whitelist:
            fwc.warn(" (number of features above may be off due to whitelist)")
        if not outcomeWithOutcomeOnly:
            assert featGetter and featGetter.featureTable, "Correlate requires a specified feature table"
        featsToYield = []
        has_wildcards = False
        if blacklist:
            for term in blacklist:
                if term[-1] == '*':
                    has_wildcards = True
        
        if whitelist:
            for term in whitelist:
                if term[-1] == '*':
                    has_wildcards = True
            if not has_wildcards:
                featsToYield = whitelist
                whitelist = None
            # else:
            #     featsToYield = whitelist ## IGNORES WILDCARDS!!!!!!!
            #     featsToYield = map(lambda x:x[:-1] if x[-1] == '*' else x, featsToYield)

        # _warn('these are the features of yield...')
        # _warn(str(featsToYield))
        if not outcomeWithOutcomeOnly:
            for (feat, dataDict, numFeats) in featGetter.yieldGroupNormsWithZerosByFeat(groups, feats = featsToYield): #switched to iter
                #Apply Whitelist, Blacklist
                if blacklist:
                    bl = False
                    bl = feat in blacklist or (has_wildcards and self.wildcardMatch(feat, blacklist))
                    for term in blacklist:
                        bl = bl or term in feat
                    if bl:
                        continue
                        
                if whitelist:
                    if not (feat in whitelist or (has_wildcards and self.wildcardMatch(feat, whitelist))):
                        #print "feat did not match: %s" % feat
                        continue
                yield (groups, allOutcomes, controls, feat, dataDict, numFeats, featFreqs)
        
        if outcomeWithOutcome or outcomeWithOutcomeOnly:
            numOutcomes = len(allOutcomes)
            for outcomeName,datadict in allOutcomes.items(): 
                yield (groups, allOutcomes, controls, 'outcome_'+outcomeName, datadict, numOutcomes, featFreqs)
    
    def IDPcomparison(self, featGetter, sample1, sample2, blacklist=None, whitelist = None):
        """
        Finds the correlations between features and outcomes

        Parameters
        ----------
        featGetter : featureGetter object
        sample1 :
        sample2 :
        blacklist : :obj:`list`, optional
            list of feature table fields (str) to ignore
        whitelist : :obj:`list`, optional
            list of feature table fields (str) to include

        Returns
        ------
        out : dict

        """
        # Applying group frequency threshold
        groups = []
        if self.group_freq_thresh:
            wordTable = self.getWordTable()
            groups = [str(i[1]) for i in mm.executeGetList(self.corpdb, self.dbCursor, "select sum(value), group_id from %s group by group_id" % wordTable, charset=self.encoding, use_unicode=self.use_unicode) if int(i[0]) >= self.group_freq_thresh]
        else:
            groups = [str(i[0]) for i in mm.executeGetList(self.corpdb, self.dbCursor, "select distinct group_id from %s" % wordTable, charset=self.encoding, use_unicode=self.use_unicode)]
            
        # Checking for group wildcards in the samples
        if sample1 == ['*']:
            sample1 = set(groups)
        elif any(['*' == g[-1] for g in sample1]):
            sample1_new = set([i for i in sample1 if '*' != i[-1]])
            for g in list(sample1)[:]:
                if g[-1] == '*':
                    sample1_new = sample1_new | set([i for i in groups if g[:-1] == i[:len(g[:-1])]])
            sample1 = sample1_new
        else:
            sample1 = set(sample1)

        if sample2 == ['*']:
            print("Using all groups")
            sample2 = set(groups)
        elif any(['*' == g[-1] for g in sample2]):
            sample2_new = set([i for i in sample2 if '*' != i[-1]])
            for g in list(sample2)[:]:
                if g[-1] == '*':
                    sample2_new = sample2_new | set([i for i in groups if g[:-1] == i[:len(g[:-1])]])
            sample2 = sample2_new
        else:
            sample2 = set(sample2)        

        if sample1 & sample2:
            if len(sample1) < len(sample2):
                fwc.warn("****** WARNING: some of the groups in sample1 are also present in sample2. Those groups will be removed from sample2 ******")
                sample2 = sample2 - sample1
            else:
                fwc.warn("****** WARNING: some of the groups in sample2 are also present in sample1. Those groups will be removed from sample1 ******")
                sample1 = sample1 - sample2

        raise NotImplementedError("Need to look at only those features that have been used by the users in the smaller sample")

        sample1, sample2 = list(sample1), list(sample2)
        print("Number of groups in Sample1: %d, Sample2: %d, Total: %d" % (len(sample1), len(sample2), len(groups)))
        
        # How many features are there? Need to chunk if too many because it'll cause a memory error
        # numFeatsQuery = "select distinct feat from %s" % featGetter.featureTable
        # numFeats = [i[0] for i in mm.executeGetList(self.corpdb, self.dbCursor, numFeatsQuery)]
        # print "Total number of features:", len(numFeats)

        values = {}
        freqsDict = {}
        for i, gs in enumerate([groups, sample1, sample2]):
            if i == 0:
                sql = "select feat, sum(value), sum(group_norm) from %s where group_id in ('%s') group by feat" % (featGetter.featureTable, "', '".join(str(i) for i in gs))
                # fill in dictionary for 1st time
                res = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
                values = {feat: [float(gn)] for feat, freq, gn in res}
                freqsDict = {feat: int(freq) for feat, freq, gn in res}
            else:
                sql = "select feat, sum(group_norm) from %s where group_id in ('%s') group by feat" % (featGetter.featureTable, "', '".join(str(i) for i in gs))
                new_values = {feat: gn for feat, gn in mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)}
                for feat, gnList in values.items():
                    gnList.append(new_values.get(feat, 0))

        results = array(('Blah', 0, 0, 0, 0.0), dtype = [('feat', 'S80'), ('1', '<f8'), ('2', '<f8'), ('all', '<f8'), ('delta', '<f8')])
        results = tile(results, (len(values),))
        results['feat'] = list(values.keys())
        counts = array([gns for feat, gns in values.items()])
        results['1'] = counts[:,1]
        results['2'] = counts[:,2]
        results['all'] = counts[:,0]
        sums = counts.sum(axis=0)

        (f1, f2) = (float(sums[1])/sums[0],float(sums[2])/sums[0])
        l1 = (counts[:,1] + f1*counts[:,0]) / ((sums[1]+f1*sums[0]) - (counts[:,1] + f1*counts[:,0]))
        l2 = (counts[:,2] + f2*counts[:,0]) / ((sums[2]+f2*sums[0]) - (counts[:,2] + f2*counts[:,0]))
        sigma = sqrt( 1.0/(counts[:,1] + f1*counts[:,0]) + 1.0/(counts[:,2]+f2*counts[:,0]))
        results['delta'] = (nplog(l2) - nplog(l1)) / sigma
        results = npsort(results,order='delta')
        endFeats = set(results['feat'])
        
        out = {}
        out['comparative_wc'] = {line[0]: (line[4], 0.0, len(groups), freqsDict[line[0]]) for line in results if line[0] in endFeats}
        return out

    def IDP_correlate(self, featGetter, outcomeWithOutcome = False, includeFreqs = False, useValuesInsteadOfGroupNorm = False, blacklist=None, whitelist = None):
        """
        Informative Dirichlet prior, based on http://pan.oxfordjournals.org/content/16/4/372.full
        Finds the correlations between features and outcomes

        Parameters
        ----------
        featGetter : featureGetter object
        outcomeWithOutcome : :obj:`boolean`, optional
            Adds the outcomes themselves to the list of variables to correlate with the outcomes if True
        includeFreqs : :obj:`boolean`, optional
            Include the frequency of each feature if True
        useValuesInsteadOfGroupNorm : :obj:`boolean`, optional
            use value field instead of group_norm
        blacklist : :obj:`list`, optional
            list of feature table fields (str) to ignore
        whitelist : :obj:`list`, optional
            list of feature table fields (str) to include
        
        Returns
        ------
        out : dict

        """
        out = dict()
        if blacklist:
            blacklist_re = list()
            for term in blacklist:
                try: 
                    blacklist_re.append(re.compile(re.sub('\*',r'\w*',term.lower())))    
                except Exception:
                    sys.stderr.write("regexp isn't valid: %s\n" % term)
        if whitelist:
            whitelist_re = list()
            for term in whitelist:
                try: 
                    whitelist_re.append(re.compile(re.sub('\*',r'\w*',term.lower())))    
                except Exception:
                    sys.stderr.write("regexp isn't valid: %s\n" % term)

        sql = "select feat, sum(value), sum(group_norm) from %s group by feat" % (featGetter.featureTable)
        counts_list = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
        
        (groups, allOutcomes, controls) = self.getGroupsAndOutcomes()
        # groups = set of group_id's that have a non null outcome (for all outcomes ever) aka is useless

        for outcome in self.outcome_value_fields:
            counts_dict = {feat: {'value': [int(value),], 'group_norm': [group_norm,]} for feat, value, group_norm in counts_list}
            outcome_groups = list(allOutcomes[outcome].keys())
            labels = sorted(set(allOutcomes[outcome].values()))
            if len(labels) > 2:
                raise ValueError("Outcome Encoding Error: IDP only works on binary outcomes")
            freqs = {}
            for value in labels:
                good_groups = [i for i in outcome_groups if allOutcomes[outcome][i]==value]
                sql = "select feat, sum(value), sum(group_norm) from %s where group_id in ('%s')" % (featGetter.featureTable, "','".join([str(i) for i in good_groups]))
                sql += " group by feat"
                value_dict = {feat : {'value': int(value), 'group_norm': group_norm} for feat, value, group_norm in mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)}
                                    
                for feat, value_group_norm_dict in counts_dict.items():
                    try:
                        value_group_norm_dict['value'].append(value_dict[feat]['value'])
                        value_group_norm_dict['group_norm'].append(value_dict[feat]['group_norm'])
                    except KeyError:
                        value_group_norm_dict['value'].append(0)
                        value_group_norm_dict['group_norm'].append(0.0)

            value_considered = 'group_norm'
            results = array(('Blah', 0, 0, 0, 0.0), dtype = [('feat', np.dtype(("U", 64))), ('1', '<f8'), ('2', '<f8'), ('all', '<f8'), ('delta', '<f8')])
            if useValuesInsteadOfGroupNorm:
                value_considered = 'value'
                results = array(('Blah', 0, 0, 0, 0.0), dtype = [('feat', np.dtype(("U", 64))), ('1', '<i8'), ('2', '<i8'), ('all', '<i8'), ('delta', '<f8')])
            results = tile(results, (len(counts_dict),))
            results['feat'] = list(counts_dict.keys())

            fwc.warn("Using '%s' to find relationship" % value_considered)
            counts = array([i[value_considered] for i in list(counts_dict.values())])
            if counts.shape[1] < 2:
                print("Your outcomes table is empty!(probably)")
                raise IndexError
            print(counts)
            results['1'] = counts[:,1]
            results['2'] = counts[:,2]
            results['all'] = counts[:,0]
            freqs = {feat: results['1'][i]+results['2'][i] for i, feat in enumerate(results['feat'])}
            sums = counts.sum(axis=0)
            (f1, f2) = (float(sums[1])/sums[0],float(sums[2])/sums[0])
            
            l1 = (counts[:,1] + f1*counts[:,0]) / ((sums[1]+f1*sums[0]) - (counts[:,1] + f1*counts[:,0]))
            l2 = (counts[:,2] + f2*counts[:,0]) / ((sums[2]+f2*sums[0]) - (counts[:,2] + f2*counts[:,0]))
            sigma = sqrt( 1.0/(counts[:,1] + f1*counts[:,0]) + 1.0/(counts[:,2]+f2*counts[:,0]))
            results['delta'] = (nplog(l2) - nplog(l1)) / sigma
            results = npsort(results,order='delta')
            endFeats = set(results['feat'])
            if blacklist:
                for feat in results['feat']:
                    for reg in blacklist_re:
                        if reg.match(feat):
                            endFeats.discard(feat)
            if whitelist:
                endFeats = set()
                for feat in results['feat']:
                    for reg in whitelist_re:
                        if reg.match(feat):
                            endFeats.add(feat)
            out[outcome] = {line[0]: (line[4], 0.0, len(outcome_groups), int(freqs[line[0]])) for line in results if line[0] in endFeats}
        
        return out

    def zScoreGroup(self, featGetter, outcomeWithOutcome = False, includeFreqs = False, blacklist=None, whitelist = None):
        """
        Calculates group zScore

        Parameters
        ----------
        featGetter : featureGetter object
        outcomeWithOutcome : :obj:`boolean`, optional
            Adds the outcomes themselves to the list of variables to correlate with the outcomes if True
        includeFreqs : :obj:`boolean`, optional
            Include the frequency of each feature if True
        blacklist : :obj:`list`, optional
            list of feature table fields (str) to ignore
        whitelist : :obj:`list`, optional
            list of feature table fields (str) to include

        Returns
        ------
        correls : dict
            dict of outcome=>feature=>(R, p, numGroups, CI, featFreqs)


        """
        correls = dict() #dict of outcome=>feature=>(R, p, numGroups, featFreqs)
        numFeatsDone = 0
        for (groups, allOutcomes, controls, feat, dataDict, numFeats, featFreqs) in self.yieldDataForOneFeatAtATime(featGetter, blacklist, whitelist, outcomeWithOutcome, includeFreqs):
            # groups: contains all groups looked at -> ie all users
            # allOutcomes: contains a dictionary of the outcomes and their values for each group in groups
            # dataDict: contains the group_norms (i.e what we're z-scoring) for every feature
            
            # This is gonne be filled with (zscorevalue, 0, numgroups, featFreq)
            tup = ()
            for outcomeField, outcomes in allOutcomes.items():
                
                labels = sorted(set(allOutcomes[outcomeField].values()))
                if len(labels) > 2:
                    print("Outcome Encoding Error: zScoreGroup only works on binary outcomes")
                    raise ValueError

                (dataList, outcomeList) = fwc.alignDictsAsLists(dataDict, outcomes)
                # Aligned lists of group_norm, outcome_value
                # To-do:
                # zscore dataList, and then return the item in the list that corresponds to the 
                # item in outcomeList that has value 1
                
                zscores = zscore(dataList)
                i = outcomeList.index(1)
                tup = (dataList[i], 0, len(outcomeList), featFreqs[feat])
                try:
                    correls[outcomeField][feat] = tup
                except KeyError:
                    correls[outcomeField] = {feat: tup}
            numFeatsDone += 1
            if numFeatsDone % 200 == 0: print("%6d features z-scored" % numFeatsDone)
        return correls

    def correlateWithFeatures(self, featGetter, spearman = False, p_correction_method = 'BH', interaction = None, 
                              blacklist=None, whitelist=None, includeFreqs = False, outcomeWithOutcome = False, outcomeWithOutcomeOnly = False, 
                              zscoreRegression = True, logisticReg = False, outputInteraction = False, groupsWhere = ''):
        """Finds the correlations between features and outcomes

        Parameters
        ----------
        featGetter : featureGetter object
        spearman : :obj:`boolean`, optional
        p_correction_method : :obj:`str`, optional
            Specified method for p-value correction
        iteraction : :obj:`list`, optional
            list of iteraction terms
        blacklist : :obj:`list`, optional
            list of feature table fields (str) to ignore
        whitelist : :obj:`list`, optional
            list of feature table fields (str) to include
        includeFreqs : :obj:`boolean`, optional
            Include the frequency of each feature if True
        outcomeWithOutcome : :obj:`boolean`, optional
            Adds the outcomes themselves to the list of variables to correlate with the outcomes if True
        outcomeWithOutcomeOnly : :obj:`boolean`, optional
            True - only correlate outcomes with outcomes
            False - correlate features with outcomes
        zscoreRegression : :obj:`boolean`, optional
            standard both variables if True
        logisticReg : :obj:`boolean`, optional
            True - use logistic regression, False - use default linear
        outputInteraction : :obj:`boolean`, optional
            True - append output interactions to results
        groupsWhere : :obj:`str`, optional
            string specified with the --where flag containing a sql statement for filtering

        Returns
        --------
        correls : dict
            dict of outcome=>feature=>(R, p, numGroups, CI, featFreqs)

        """
        
        correls = dict() #dict of outcome=>feature=>(R, p, numGroups, (conf_int), featFreqs)
        numRed = 0
        
        firstLoop = True
        #print "WHITELIST: %s" % str(whitelist) #debug

        # Yields data for one feature at a time
        for (groups, allOutcomes, controls, feat, dataDict, numFeats, featFreqs) in self.yieldDataForOneFeatAtATime(featGetter, blacklist, whitelist, outcomeWithOutcome, includeFreqs, groupsWhere, outcomeWithOutcomeOnly):
            # Looping over outcomes
            for outcomeField, outcomes in allOutcomes.items() :
                tup = ()
                interaction_tuples = {}

                # find correlation or regression coef, p-value, and N (stored in tup)
                if controls or logisticReg: #run OLS or logistic regression

                    if firstLoop and controls:
                        # Do regression showing the effect of the controls only
                        # i.e. show the coefficients from the controls alone
                        
                        (X, y) = fwc.alignDictsAsXy([controls[k] for k in sorted(controls.keys())], outcomes)
                        #X = np.array(X).astype(np.float)#debug: we should be able to avoid this by casting correctly originally
                        #y = np.array(y).astype(np.float)#debug: we should be able to avoid this by casting correctly originally
 
                        # print "alignDict time: %f"% float(time.time() - t0)#debug
                        if spearman:
                            X = fwc.switchColumnsAndRows([rankdata(x)
                                                      for x in fwc.switchColumnsAndRows(X)])
                            y = rankdata(y)
                        if zscoreRegression: 
                            # print "MAARTEN\t", type(X[0][0]), type(y[0])
                            (X, y) = (zscore(X), zscore(y) if not logisticReg else y)
                        results = None
                        try:
                            if logisticReg:
                                results = sm.Logit(y, X).fit(disp=False) #runs regression
                            else:
                                results = sm.OLS(y, X).fit() #runs regression
                            conf = fwc.conf_interval(results.params[-1], len(y))
                            tup = (results.params[-1], results.pvalues[-1], len(y), conf)

                            print(results.summary(outcomeField, sorted(controls.keys())))#debug
                        except (ValueError, Exception) as err:
                            mode = 'Logistic regression' if logisticReg else 'OLS'
                            fwc.warn("%s threw ValueError: %s" % (mode,str(err)))
                            fwc.warn(" feature '%s' with outcome '%s' results not included" % (feat, outcomeField))
                    #t0 = time.time()#debug

                    # Interaction: append to X the multiplication of outcome column & the interaction
                    
                    controlsValues = [values for control, values in controls.items() if control not in interaction] + [controls[i] for i in interaction]
                    controlsKeys = [control for control, values in controls.items() if control not in interaction] + interaction

                    # controls.values() makes the labels go away, turns it into a list
                    (X, y) = fwc.alignDictsAsXy(controlsValues + [dataDict], outcomes)
                    # X is a matrix, y is a column vector
                    # Each row of X is: [control1, control2, ..., interaction1, interaction2, ..., group_norm]
                    #                       0         1           len(controls) len(controls)+1    len(controls)+len(interaction)

                    for i in range(len(controls)-len(interaction), len(controls)):
                        X = [x[:-1]+[float(x[i])*x[-1], x[-1]] for x in X]

                    # X is a matrix, y is a column vector
                    # Each row of X is: [control1, control2, ..., interaction1,       interaction2, ..., interaction1*group_norm, int2*gn, ..., group_norm]
                    #                       0         1        len(ctrls)-len(int)                              len(ctrls)                 len(controls)+len(interaction)
                    
                    #print "alignDict time: %f"% float(time.time() - t0)#debug
                    if spearman: 
                        X = fwc.switchColumnsAndRows([rankdata(x) for x in fwc.switchColumnsAndRows(X)])
                        y = rankdata(y)
                    #X = np.array(X).astype(np.float)#debug: we should be able to avoid this by casting correctly originally
                    #y = np.array(y).astype(np.float)#debug: we should be able to avoid this by casting correctly originally

                    if zscoreRegression: (X, y) = (zscore(X), zscore(y) if not logisticReg else y)

                    results = None
                    try:
                        if logisticReg:
                            results = sm.Logit(y, X, missing='drop').fit(disp=False)
                        else:
                            results = sm.OLS(y, X).fit() #runs regression
                        conf = fwc.conf_interval(results.params[-1], len(y))
                        tup = (results.params[-1], results.pvalues[-1], len(y), conf)

                        if outputInteraction:
                            interaction_tuples = {}
                            for i, inter in enumerate(interaction):
                                interaction_tuples["%s with %s" % (inter, outcomeField)] = (results.params[i+len(controls)-len(interaction)], results.pvalues[i+len(controls)-len(interaction)], len(y))
                                interaction_tuples["group_norm * %s from %s" % (inter, outcomeField)] = (results.params[i+len(controls)], results.pvalues[i+len(controls)], len(y))
                            
                    except (ValueError,Exception) as err:
                        mode = 'Logistic regression' if logisticReg else 'OLS'
                        fwc.warn("%s threw ValueError: [%s]" % (mode, str(err)))
                        fwc.warn(" feature '%s' with outcome '%s' results not included" % (feat, outcomeField))

                else: #run pearson / spearman correlation (if not logitsic or not controls)
                    (dataList, outcomeList) = fwc.alignDictsAsLists(dataDict, outcomes)
                    # pdb.set_trace()
                    #LAD addition: added because pearsonr messes up when trying to regress between different types: Decimal and float
                    outcomeList = list(map(float, outcomeList))
                             
                    if spearman: tup = spearmanr(dataList, outcomeList) + (len(dataList),)
                    else: tup = pearsonr(dataList, outcomeList) + (len(dataList),)
                    conf = fwc.conf_interval(tup[self.r_idx], tup[self.n_idx])
                    tup = tup + (conf,)
                if not tup or not tup[self.r_idx]:
                    fwc.warn("unable to correlate feature '%s' with '%s'" %(feat, outcomeField))
                    if includeFreqs: tup = (float('nan'), float('nan'), len(y), (float('nan'), float('nan')), 0)
                    else: tup = (float('nan'), float('nan'), len(y), (float('nan'), float('nan')))
                else: 
                    if p_correction_method.startswith("bonf"):
                        tup = fwc.bonfPCorrection(tup, numFeats)
                        if outputInteraction: interaction_tuples = {k: fwc.bonfPCorrection(v, numFeats) + v[2:] for k, v in interaction_tuples.items()} 
                    if includeFreqs:
                        try:
                            tup = tup + (int(featFreqs[feat]), )
                            if outputInteraction: 
                                if self.use_unicode:
                                    interaction_tuples = {k: v + (int(featFreqs[str(feat)]), ) for k, v in interaction_tuples.items()} 
                                else:
                                    interaction_tuples = {k: v + (int(featFreqs[feat]), ) for k, v in interaction_tuples.items()} 
                            
                        except KeyError:
                            if not whitelist:
                                fwc.warn("unable to find total freq for '%s'" % feat)
                            tup = tup + (float('nan'), )
                            if outputInteraction:
                                interaction_tuples = {k: v + (float('nan'), ) for k, v in interaction_tuples.items()} 
                try: 
                    correls[outcomeField][feat] = tup
                except KeyError:
                    correls[outcomeField] = {feat: tup}
                if outputInteraction:
                    for inter, tup in interaction_tuples.items():
                        try:
                            correls[inter][feat] = tup
                        except KeyError:
                            correls[inter] = {feat: tup}

            numRed += 1
            if numRed % 200 == 0: fwc.warn("  %d features correlated"%(numRed))
            firstLoop = False
        # exit() # Maarten

        if p_correction_method and not p_correction_method.startswith("bonf"): 
            ##change correls here. 
            for outcomeField, featCorrels in correls.items():
                pDict = dict( [(k, tup[self.p_idx]) for k, tup in featCorrels.items()] ) 
                rDict = dict( [(k, tup[self.r_idx]) for k, tup in featCorrels.items()] ) 
                pDict = fwc.pCorrection(pDict, p_correction_method, [0.05, 0.01, 0.001], rDict = rDict)
                for k, tup in featCorrels.items():
                    featCorrels[k] = (tup[self.r_idx], pDict[k]) + tup[self.n_idx:]

        if self.featureMapping:
            newCorrels = dict()
            for outcomeField, featRs in correls.items():
                newCorrels[outcomeField] = dict()
                for feat in featRs:
                    newCorrels[outcomeField][self.mapFeatureName(feat, self.featureMapping)] = featRs[feat]
            correls = newCorrels 

        return correls

    def aucWithFeatures(self, featGetter, p_correction_method = 'BH', interaction = None, bootstrapP = None, blacklist=None, 
                        whitelist=None, includeFreqs = False, outcomeWithOutcome = False, zscoreRegression = True, outputInteraction = False, groupsWhere = ''):
        """

        Finds the auc between features and dichotamous outcomes

        Parameters
        ----------

        featGetter : featureGetter object
        p_correction_method : :obj:`str`, optional
            Specified method for p-value correction
        iteraction : :obj:`list`, optional
            list of iteraction terms
        bootstrapP : :obj:`list`, optional
        blacklist : :obj:`list`, optional
            list of feature table fields (str) to ignore
        whitelist : :obj:`list`, optional
            list of feature table fields (str) to include
        includeFreqs : :obj:`boolean`, optional
            Include the frequency of each feature if True
        outcomeWithOutcome ::obj:`boolean`, optional
            Adds the outcomes themselves to the list of variables to correlate with the outcomes if True
        zscoreRegression : :obj:`boolean`, optional
            standard both variables if True
        outputInteraction : :obj:`boolean`, optional
            True - append output interactions to results
        groupsWhere : :obj:`str`, optional
            string specified with the --where flag containing a sql statement for filtering

        Returns
        -------
        aucs : dict
            dict of outcome=>feature=>(auc, p, numGroups, featFreqs)

        """
        
        aucs = dict() #dict of outcome=>feature=>(auc, p, numGroups, featFreqs)
        numRed = 0
        
        firstLoop = True
        featNum = 0
        #print "WHITELIST: %s" % str(whitelist) #debug

        # Yields data for one feature at a time
        for (groups, allOutcomes, controls, feat, dataDict, numFeats, featFreqs) in self.yieldDataForOneFeatAtATime(featGetter, blacklist, whitelist, outcomeWithOutcome, includeFreqs, groupsWhere):
            # Looping over outcomes
            featNum += 1
            for outcomeField, outcomes in allOutcomes.items() :
                tup = ()

                # find correlation or regression coef, p-value, and N (stored in tup)
                if controls: #consider controls

                    (X, y) = fwc.alignDictsAsXy([controls[k] for k in sorted(controls.keys())] + [dataDict], outcomes)
                    if zscoreRegression: 
                        X = zscore(X)

                    if firstLoop and controls:
                        # Do regression showing the effect of the controls only
                        # i.e. show the coefficients from the controls alone
                        
                        print("\n= %11s == AUC WITH CONTROLS =\n=================================" % outcomeField)
                        auc = None
                        try:
                            #print "  RANDOM TEST: %.4f" % mean([roc_auc_score(y , [randint(0,500)/100.0 for i in range(655)]) for i in xrange(100)])
                            for cntrl, i in zip(sorted(controls.keys()), list(range(len(list(controls.keys()))))):
                                auc = roc_auc_score(y, X[:,i])
                                if auc < 0.5:
                                    auc -= 1
                                print("  %11s: %.4f" %(cntrl, auc))
                        except (ValueError, Exception) as err:
                            fwc.warn("threw ValueError: %s" % str(err))
                            fwc.warn(" feature '%s' with outcome '%s' results not included" % (feat, outcomeField))
                            #TODO: add line of error
                        #all controls alone:
                        lr = LogisticRegression(penalty='l2', C=1000000, fit_intercept=True)
                        Xc = X[:,:-1]
                        probs = lr.fit(Xc,y).predict_proba(Xc)
                        cauc = roc_auc_score(y, probs[:,1])
                        print(" ALL CONTROLS: %.4f" % cauc)
                        print("===================================\n")

                    #do the rest:
                    lr = LogisticRegression(penalty='l2', C=1000000, fit_intercept=True)
                    probs = lr.fit(X,y).predict_proba(X)
                    auc = roc_auc_score(y, probs[:,1])
                    if lr.coef_[0,-1] < 0: #mark the auc negative if the relationship with the feature is neg
                        auc = -1 * auc
                    if bootstrapP:
                        check = abs(auc)
                        if check > (abs(cauc) + .01):
                            print("%d/%d: %.3f cauc vs %.3f c+tpc (%.3f difference); YES bootstrapping" % (featNum,numFeats,abs(cauc),check,check-abs(cauc)))
                            Xc = X[:,:-1]
                            Xend = X[:,-1][...,None]
                            pool = multiprocessing.Pool(int(fwc.CORES/3))        
                            fCount = sum(pool.map(fwc.fiftyChecks, [(Xc, Xend, y, check)]*int(bootstrapP/50) ) ) 

                            tup = (auc, fCount/float(bootstrapP), len(y))
                            pool.close()
                        else:
                            print("%d/%d: %.3f cauc vs %.3f c+tpc (%.3f difference); NO bootstrapping" % (featNum,numFeats,abs(cauc),check,check-abs(cauc)))
                            tup = (auc, 1.0, len(y))
                    else:
                        tup = (auc, 0.0, len(y))

                                        
                else: #no controls
                    cauc = 0.50
                    (X, y) = fwc.alignDictsAsLists(dataDict, outcomes)
                    y = list(map(float, y))
                    if zscoreRegression:
                        X = zscore(X)
                    try:
                        auc = roc_auc_score(y, X)
                        if auc < 0.5:
                            auc -= 1
                    except (ValueError, Exception) as err:
                        fwc.warn("threw ValueError: %s" % str(err))
                        fwc.warn(" feature '%s' with outcome '%s' results not included" % (feat, outcomeField))

                    #Bootstrap without controls:
                    if bootstrapP:
                        check = abs(auc)
                        if check > (abs(cauc) + .01):
                            print("%d/%d: %.3f cauc vs %.3f c+tpc (%.3f difference); YES bootstrapping" % (featNum,numFeats,abs(cauc),check,check-abs(cauc)))
                            #print X
                            Xc = None
                            #Xend = X[:,-1][...,None]
                            pool = multiprocessing.Pool(int(fwc.CORES/3))
                            fCount = sum(pool.map(fwc.fiftyChecks, [(Xc, X, y, check)]*int(bootstrapP/50) ) )
                            # print fCount
                            # test = fiftyChecks((Xc, X, y, check))
                            # print test
                            # sys.exit()
                            tup = (auc, fCount/float(bootstrapP), len(y))
                            pool.close()
                        else:
                            print("%d/%d: %.3f cauc vs %.3f c+tpc (%.3f difference); NO bootstrapping" % (featNum,numFeats,abs(cauc),check,check-abs(cauc)))
                            tup = (auc, 1.0, len(y))
                    else:
                        tup = (auc, 0.0, len(y))

                #adjust or add to tup...
                if not tup or not tup[0]:
                    fwc.warn("unable to AUC feature '%s' with '%s'" %(feat, outcomeField))
                    if includeFreqs: tup = (float('nan'), float('nan'), len(y), 0)
                    else: tup = (float('nan'), float('nan'), len(y))
                else: 
                    if p_correction_method.startswith("bonf"):
                        tup = fwc.bonfPCorrection(tup, numFeats)
                    if includeFreqs:
                        try:
                            if self.use_unicode:
                                tup = tup + (int(featFreqs[str(feat)]), )
                            else:
                                tup = tup + (int(featFreqs[feat]), )

                        except KeyError:
                            if not whitelist:
                                fwc.warn("unable to find total freq for '%s'" % feat)
                            tup = tup + (float('nan'), )
                    try: 
                        aucs[outcomeField][feat] = tup
                    except KeyError:
                        aucs[outcomeField] = {feat: tup}

                numRed += 1
                if numRed % 200 == 0: fwc.warn("  %d features correlated"%(numRed))
                firstLoop = False

        if self.featureMapping:
            #pickle.dump((aucs, self.featureMapping), open("aucs-featmapping.p", "wb"))
            newAucs = dict()
            for outcomeField, featRs in aucs.items():
                newAucs[outcomeField] = dict()
                for feat in featRs:
                    newAucs[outcomeField][self.mapFeatureName(feat, self.featureMapping)] = featRs[feat]
            aucs = newAucs 

        return aucs


    def correlateControlCombosWithFeatures(self, featGetter, spearman = False, p_correction_method = 'BH', 
                              blacklist=None, whitelist=None, includeFreqs = False, outcomeWithOutcome = False, zscoreRegression = True):
        """
        Finds the correlations between features and all combinations of outcomes

        Parameters
        ----------
        featGetter : featureGetter object
        spearman : :obj:`boolean`, optional
        p_correction_method : :obj:`str`, optional
            Specified method for p-value correction
        blacklist : :obj:`list`, optional
            list of feature table fields (str) to ignore
        whitelist : :obj:`list`, optional
            list of feature table fields (str) to include
        includeFreqs : :obj:`boolean`, optional
            Include the frequency of each feature if True
        outcomeWithOutcome : :obj:`boolean`, optional
            Adds the outcomes themselves to the list of variables to correlate with the outcomes if True
        zscoreRegression : :obj:`boolean`, optional
            standard both variables if True

        Returns
        -------
        comboCorrels : dict
            dict of outcome=>feature=>(R, p)

        """

        comboCorrels = dict() #dict of outcome=>feature=>(R, p)
        numRed = 0
        firstLoop = True

        for (groups, allOutcomes, controls, feat, dataDict, numFeats, featFreqs) in self.yieldDataForOneFeatAtATime(featGetter, blacklist, whitelist, outcomeWithOutcome, includeFreqs):

            controlKeys = list(allControls.keys())
            for r in range(len(controlKeys)+1):
                for controlKeyCombo in combinations(controlKeys, r):
                    controls = dict()
                    if len(controlKeyCombo) > 0:
                        controls = dict([(k, allControls[k]) for k in controlKeyCombo])
                    controlKeyCombo = tuple(controlKeyCombo)
                    for outcomeField, outcomes in allOutcomes.items():
                        tup = ()

                        #find correlation or regression coef, p-value, and N (stored in tup)
                        if controls:
                            # If controls: run OLS regression
                            
                            if firstLoop:
                                # show the coefficients from the controls alone
                                thisControlKeys = sorted(controls.keys())
                                (X, y) = fwc.alignDictsAsXy([controls[k] for k in thisControlKeys], outcomes)
                                
                                # print "alignDict time: %f"% float(time.time() - t0)#debug
                                
                                if spearman: 
                                    X = fwc.switchColumnsAndRows([rankdata(x) for x in fwc.switchColumnsAndRows(X)])
                                    y = rankdata(y)
                                if zscoreRegression: (X, y) = (zscore(X), zscore(y))
                                
                                results = None
                                try:
                                    
                                    # run regression
                                    results = sm.OLS(y, X).fit()
                                                                        
                                    try: 
                                        comboCorrels[outcomeField][controlKeyCombo] = dict()
                                    except KeyError:
                                        comboCorrels[outcomeField] = {controlKeyCombo : dict()}
                                    for c in range(len(results.params)):
                                        tup = (results.params[c], results.pvalues[c], len(y))
                                        comboCorrels[outcomeField][controlKeyCombo]['__CONTROL_'+thisControlKeys[c]] = tup
                                    print(results.summary(outcomeField, sorted(controls.keys())))#debug
                                except ValueError as err:
                                    fwc.warn("OLS threw ValueError: %s" % str(err))
                                    fwc.warn(" feature '%s' with outcome '%s' results not included" % (feat, outcomeField))

                            #t0 = time.time()#debug
                            
                            (X, y) = fwc.alignDictsAsXy(list(controls.values()) + [dataDict], outcomes)
                            
                            # print "alignDict time: %f"% float(time.time() - t0)#debug
                            
                            if spearman: 
                                X = fwc.switchColumnsAndRows([rankdata(x) for x in fwc.switchColumnsAndRows(X)])
                                y = rankdata(y)
                            if zscoreRegression: (X, y) = (zscore(X), zscore(y))
                            results = None
                            try:
                                results = sm.OLS(y, X).fit() #runs regression
                                tup = (results.params[-1], results.pvalues[-1], len(y))
                                #print results.summary(outcomeField, controls.keys()+[feat])#debug
                            except ValueError as err:
                                fwc.warn("OLS threw ValueError: %s" % str(err))
                                fwc.warn(" feature '%s' with outcome '%s' results not included" % (feat, outcomeField))

                        else:
                            # If not controls : run pearson / spearman correlation
                            (dataList, outcomeList) = fwc.alignDictsAsLists(dataDict, outcomes)
                            if spearman: tup = spearmanr(dataList, outcomeList) + (len(dataList),)
                            else: tup = pearsonr(dataList, outcomeList) + (len(dataList),)

                        if not tup or not tup[0]:
                            fwc.warn("unable to correlate feature '%s' with '%s'" %(feat, outcomeField))
                            if includeFreqs: tup = (float('nan'), float('nan'), float('nan'), float('nan'))
                            else: tup = (float('nan'), float('nan'), float('nan'))
                        else: 
                            if p_correction_method.startswith("bonf"):
                                tup = fwc.bonfPCorrection(tup, numFeats)
                            if includeFreqs:

                                try:
                                    tup = tup + (int(featFreqs[str(feat)]), )
                                except KeyError:
                                    fwc.warn("unable to find total freq for '%s'" % feat)
                                    tup = tup + (float('nan'), )
                        try: 
                            comboCorrels[outcomeField][controlKeyCombo][feat] = tup
                        except KeyError:
                            try: 
                                comboCorrels[outcomeField][controlKeyCombo] = dict()
                                comboCorrels[outcomeField][controlKeyCombo][feat] = tup
                            except KeyError:
                                comboCorrels[outcomeField] = {controlKeyCombo: dict()}
                                comboCorrels[outcomeField][controlKeyCombo][feat] = tup

            numRed += 1
            if numRed % 200 == 0: fwc.warn("  %d features correlated"%(numRed))
            firstLoop = False

        

        if p_correction_method and not p_correction_method.startswith("bonf"): 
            ##change comboCorrels here. 
            for outcomeField, featComboCorrels in comboCorrels.items():
                for controlCombo, featCorrels in featComboCorrels.items():
                    pDict = dict( [(k, tup[1]) for k, tup in featCorrels.items()] ) 
                    rDict = dict( [(k, tup[0]) for k, tup in featCorrels.items()] ) 
                    pDict = fwc.pCorrection(pDict, p_correction_method, [0.05, 0.01, 0.001], rDict = rDict)
                    for k, tup in featComboCorrels.items():
                        featComboCorrels[k][controlCombo] = (tup[0], pDict[k]) + tup[2:]

        if self.featureMapping:
            #pickle.dump((comboCorrels, self.featureMapping), open("comboCorrels-featmapping.p", "wb"))
            newComboCorrels = dict()
            for outcomeField, comboFeatRs in comboCorrels.items():
                newComboCorrels[outcomeField] = dict()
                for controlCombo, featRs in comboFeatRs.items():
                    newComboCorrels[outcomeField][controlCombo] = dict()
                    for feat in featRs:
                        newComboCorrels[outcomeField][controlCombo][self.mapFeatureName(feat, self.featureMapping)] = featRs[feat]
            comboCorrels = newComboCorrels 

        return comboCorrels





    def multRegressionWithFeatures(self, featGetter, spearman = False, p_correction_method = 'BH', 
                              blacklist=None, whitelist=None, includeFreqs = False, outcomeWithOutcome = False, zscoreRegression = True, interactions = False):
        """

        Finds the multiple regression coefficient between outcomes and features

        Parameters
        ----------
        featGetter : featureGetter object
        spearman : :obj:`boolean`, optional
        p_correction_method " str
            Specified method for p-value correction
        blacklist : :obj:`list`, optional
            list of feature table fields (str) to ignore
        whitelist : :obj:`list`, optional
            list of feature table fields (str) to include
        includeFreqs : :obj:`boolean`, optional
            Include the frequency of each feature if True
        outcomeWithOutcome : :obj:`boolean`, optional
            Adds the outcomes themselves to the list of variables to correlate with the outcomes if True
        zscoreRegression : :obj:`boolean`, optional
            standard both variables if True
        iteractions : :obj:`boolean`, optional

        Returns
        -------
        coeffs : dict
            dict of feature=>outcome=>(R, p)

        """
        #outcomes => things to find coefficients for
        #controls => included as covariates but coefficients not stored
        #language feature => y in regression

        coeffs = dict() #dict of feature=>outcome=>(R, p)
        numRed = 0
        previousGroupSetting = self.oneGroupSetForAllOutcomes
        self.oneGroupSetForAllOutcomes = True #makes sure the groups returned are the intersection of data available
        for (groups, allOutcomes, controls, feat, dataDict, numFeats, featFreqs) in self.yieldDataForOneFeatAtATime(featGetter, blacklist, whitelist, outcomeWithOutcome, includeFreqs):

            if not controls: controls = dict()
            currentCoeffs = dict() #outcome_name => coeff

            #find regression coef, p-value, and N (stored in tup)
            outcomeKeys = list(allOutcomes.keys())
            numOutcomes = len(outcomeKeys)
            outcomeColumns = dict((outcomeKeys[i], i) for i in range(len(outcomeKeys)))
            (X, y) = fwc.alignDictsAsXy(list(allOutcomes.values()) + list(controls.values()), dataDict)
            if spearman:
                X = fwc.switchColumnsAndRows([rankdata(x) for x in fwc.switchColumnsAndRows(X)])
                y = rankdata(y)
            if zscoreRegression: (X, y) = (zscore(X), zscore(y))
            results = None
            if interactions:
                #print "INTERACTIONS" #debug
                for i in range(numOutcomes - 1):
                    for j in range(i+1, numOutcomes):
                        #add a row to X that is xi X xj
                        newColumn = multiply(X[:,i], X[:,j])
                        if zscoreRegression: newColumn = zscore(newColumn)
                        npappend(X, array([newColumn]).T, 1)
                        newOutcomeName = outcomeKeys[i]+'##'+outcomeKeys[j]
                        #print "added %s" % newOutcomeName #debug
                        outcomeColumns[newOutcomeName] = X.shape[1] - 1
                        
            try:
                results = sm.OLS(y, X).fit() #runs regression
                #print results.summary(feat, allOutcomes.keys()+controls.keys())#debug
                for outcomeName, i in outcomeColumns.items():
                    tup = (results.params[i], results.pvalues[i], len(y))
                    i += 1
                    if not tup[0]: #if there was no coefficient for some reason
                        if not whitelist:
                            fwc.warn("unable to correlate feature '%s' with '%s'" %(feat, outcomeField))
                        if includeFreqs: tup = (float('nan'), float('nan'), float('nan'), float('nan'))
                        else: tup = (float('nan'), float('nan'), float('nan'))
                    else: 
                        if p_correction_method.startswith("bonf"):
                            tup = fwc.bonfPCorrection(tup, numFeats)
                        if includeFreqs:
                            try:
                                if not featFreqs: raise KeyError
                                tup = tup + (int(featFreqs[str(feat)]), )
                            except KeyError:
                                if not whitelist:
                                    fwc.warn("unable to find total freq for '%s'" % feat)
                                tup = tup + (float('nan'), )
                    currentCoeffs[outcomeName] = tup
            except ValueError as err: #if OLS couldn't run
                fwc.warn("OLS threw ValueError: %s" % str(err))
                fwc.warn(" feature '%s' with outcome '%s' results not included" % (feat, outcomeField))

            coeffs[feat] = currentCoeffs
            numRed += 1
            if numRed % 200 == 0: fwc.warn("  %d features regressed over"%(numRed))
        
        if p_correction_method and not p_correction_method.startswith("bonf"): 
            ##change correls here. 
            for feat, outcomeCoeffs in coeffs.items():
                pDict = dict( [(k, tup[1]) for k, tup in featCorrels.items()] ) 
                rDict = dict( [(k, tup[0]) for k, tup in featCorrels.items()] ) 
                pDict = fwc.pCorrection(pDict, p_correction_method, [0.05, 0.01, 0.001], rDict = rDict)
                for k, tup in outcomeCoeffs.items():
                    outcomeCoeffsCorrels[k] = (tup[0], pDict[k]) + tup[2:]

        if self.featureMapping:
            newCoeffs = dict()
            for feat, values in coeffs.items():
                newCoeffs[self.mapFeatureName(feat, self.featureMapping)] = values
            coeffs = newCoeffs

        self.oneGroupSetForAllOutcomes = previousGroupSetting #restore previous setting
        return coeffs


    def loessPlotFeaturesByOutcome(self, featGetter, spearman = False, p_correction_method = 'BH', blacklist=None, whitelist=None, 
                                   includeFreqs = False, zscoreRegression = True, outputdir='/data/ml/fb20', outputname='loess.jpg', topicLexicon=None, numTopicTerms=8, outputOrder = []):
        """
        Finds the correlations between features and outcomes

        Parameters
        ----------
        featGetter : featureGetter object
        spearman : :obj:`boolean`, optional
        p_correction_method : :obj:`str`, optional
            Specified method for p-value correction
        blacklist : :obj:`list`, optional
            list of feature table fields (str) to ignore
        whitelist : :obj:`list`, optional
            list of feature table fields (str) to include
        includeFreqs : :obj:`boolean`, optional
            Include the frequency of each feature if True
        zscoreRegression : :obj:`boolean`, optional
            standard both variables if True
        outputdir : :obj:`str`, optional
            directory for results to be written
        outputname : :obj:`str`, optional
            name of output file
        topicLexicon : :obj:`list`, optional
        numTopicTerms : :obj:`int`, optional
        outputOrder : :obj:`list`, optional

        Returns
        -------
        no return value, creates loess plot and saves it to a file

        """
        
        plotData = dict()#outcomeField->feat->data

        for (groups, allOutcomes, allControls, feat, dataDict, numFeats, featFreqs) in self.yieldDataForOneFeatAtATime(featGetter, blacklist=[], whitelist=whitelist, includeFreqs=includeFreqs):
            for outcomeField, outcomes in allOutcomes.items():
                print("Generating plot data for feature '%s' and outcome %s (controlled for %s)" % (feat, outcomeField, str(list(allControls.keys()))))
                (X, y) = (None, None)
                if allControls:
                    (X, y) = fwc.alignDictsAsXy([outcomes] + list(allControls.values()), dataDict)
                else:
                    (X, y) = fwc.alignDictsAsLists(outcomes, dataDict)
                if zscoreRegression: 
                    # (X, y) = (zscore(X), y)
                    # (X, y) = (zscore(X), zscore(y))
                    #(X, y) = (zscore(X), zscore(y))
                    (X, y) = fwc.stratifiedZScoreybyX0(X, y)
                elif allControls: fwc.warn("running loess with controls without zscore does not produce reliable output")
                sortedX0set = []
                if len(X.shape) > 1: sortedX0set = sorted(set(X[:,0]))
                else: sortedX0set = sorted(set(X))
                zscoreMap = dict(list(zip(sortedX0set, sorted(set(outcomes.values())))))
                predictX = array([sortedX0set]).T
                if len(X.shape) > 1:
                    for i in range(1, X.shape[1]):
                        meanControl = array([[mean(list(set(X[:,i])))] * predictX.shape[0]]).T
                        predictX = npappend(predictX, meanControl, axis=1)
                try: 
                    plotData[outcomeField][feat] = {'X': X, 'y': y, 'zscoreMap': zscoreMap, 'Xlabels': [outcomeField]+list(allControls.keys()), 'predictX': predictX}
                except KeyError:
                    plotData[outcomeField] = dict()
                    plotData[outcomeField][feat] = {'X': X, 'y': y, 'zscoreMap': zscoreMap, 'Xlabels': [outcomeField]+list(allControls.keys()), 'predictX': predictX}

        #plotData now contains: outcomeField->feat->(X, y)
        #X is outcomes plus controls, y is group norms (we are trying to predict group norms)
        #the first column of X is the outcome, other columns are controls
        #everything is already standardized (zscored)
        # pprint(plotData) #debug

        import uuid
        try:
            import rpy2.robjects as ro
            from rpy2.robjects.packages import importr
        except:
            fwc.warn("You must have rpy2 installed for this.")
            exit()
        
        ro.r.library('ggplot2')
        grdevices = importr('grDevices')
        # ro.r('plot = ggplot()')
        for outcome, featToFeatData in plotData.items():
            ro.r('full.plotdata <- data.frame(feature=c(), %s=c(), yhat=c(), lty=c())'%(outcome,))

            ii = 0
            featLabels = []
            for feat, featDict in featToFeatData.items():
                ii += 1
                X = featDict['X']
                y = featDict['y']
                xLabels = featDict['Xlabels']
                zscoreMap = featDict['zscoreMap']
                predictX = featDict['predictX']
                
                df = {'y':ro.FloatVector(y)}
                lenX = 1
                if len(X.shape) > 1:
                    lenX = X.shape[1]
                for ii_col in range(lenX):
                    colname = xLabels[ii_col]
                    coldata = None
                    coldata = ro.FloatVector(X[:, ii_col]) if lenX > 1 else ro.FloatVector(X)
                    df[colname] = coldata

                dfControlled = {}
                # dfControlled = {'y':ro.FloatVector(y)}
                length_cols = None
                for ii_col in range(lenX):
                    colname = xLabels[ii_col]
                    coldata = ro.FloatVector(predictX[:, ii_col])
                    dfControlled[colname] = coldata
                    length_cols = len(predictX[:, ii_col])
                dfControlled.update({'y':[0.0]*length_cols})

                rdf = ro.DataFrame(df)
                ro.globalenv['df'] = rdf

                rdfControlled = ro.DataFrame(dfControlled)
                ro.globalenv['dfControlled'] = rdfControlled

                formula = 'y ~ %s'%('+'.join(xLabels),)
                ro.r('model <- loess(%s, df, degree=1, span=.85)'%(formula,))
                # ro.globalenv['%s'%(outcome,)] = ro.FloatVector(map(lambda x: zscoreMap[x], list(X[:,0])))
                ro.globalenv['%s'%(outcome,)] = ro.FloatVector(sorted(zscoreMap.values()))

                # ro.r('''save(df, file='df.RObj')''')
                # ro.r('''save(dfControlled, file='dfControlled.RObj')''')
                # ro.r('''save(%s, file='outcome.RObj')'''%(outcome,))

                if topicLexicon:
                    featLabels.append(self.getTopicFeatLabel(topicLexicon, feat, numTopicTerms))

                ro.r('yhat <- predict(model, dfControlled)')
                # ro.r('yhat <- scale(yhat)')
                # ro.r('yhat <- (yhat - min(yhat)) / (max(yhat) - min(yhat))') ## <- LAD
                # ro.r('plotdata <- data.frame(feature=rep("%s", length(yhat)), %s=%s, yhat=yhat, lty=rep(1+floor((%d-1)/3.0), length(yhat)))'%(feat, outcome, outcome, ii)) ## <- LAD
                ro.r('plotdata <- data.frame(feature=rep("%s", length(yhat)), %s=%s, yhat=yhat)'%(feat, outcome, outcome))
                #ro.r('plotdata <- data.frame(feature=rep("%s", length(yhat)), %s=%s, yhat=yhat, lty=rep(1+floor((%d-1)/3.0), length(yhat)))'%(feat, outcome, outcome, ii)) #dashed <- HAS
                ro.r('full.plotdata <- rbind(full.plotdata, plotdata)')
                fwc.warn('finished feature: %s'%(feat,))
 
                # ro.r('''plot <- plot + scale_y_continuous('Standardized Relative Frequency', limits=c(-3.3,3.3), breaks=c(-3,0,3)) + scale_x_continuous('Age', limits=c(10, 70), breaks=seq(15,65,10))''')
                # ro.r('''save(full.plotdata, file='plotdata.RObj')''')
                time_uuid = str(uuid.uuid1())
                plotdata_filename = '/tmp/plotdata-%s.RObj'%(time_uuid,)
                ro.r('save(full.plotdata, file="%s")'%(plotdata_filename,))

                # outputname = outcome.upper() + '-' + '-'.join(featToFeatData.keys()) + '.pdf' if not outputname else outputname ## LAD
                # outputname = outcome.upper() + '-' + '-'.join(featToFeatData.keys()) + '.jpg' if not outputname else outputname
                outputname = outcome.upper() + '-' + '-'.join(list(featToFeatData.keys())) + '.svg' if not outputname else outputname ## HAS
                output_filename = os.path.join(outputdir, outputname)
                fwc.warn( 'Writing to file: %s'%(output_filename,) )

                plotscript_filename = '/tmp/plotscript-%s.R'%(time_uuid,)
                with open(plotscript_filename, 'w') as f:
                    r_commands = self.getGGplotCommands(outcome, plotdata_filename, output_filename, featLabels)
                    f.writelines(r_commands)

                ro.r('source("%s")'%(plotscript_filename,))

                os.system('rm %s'%(plotdata_filename))
                os.system('rm %s'%(plotscript_filename))

    def getTopicFeatLabel(self, topicLexicon, feat, numTopicTerms=8):
        """
        Get the terms with the highest weight for a given topicLexicon

        Parameters
        ----------
        topicLexicon : str
            table name containing the topic lexicon
        feat : str
            the topic label
        numTopicTerms : :obj:`int`, optional
            number of topic terms


        Returns
        -------
        label : list
            list of terms with highest weight from the topic lexicon for the given topic

        """
        
        sql = 'SELECT term, weight from %s WHERE category = \'%s\''%(topicLexicon, feat)
        rows = mm.qExecuteGetList('permaLexicon', sql)
        top_n_rows = sorted(rows, key=lambda x:x[1], reverse=True)
        terms = [x[0] for x in top_n_rows]
        label = ' '.join(map(str, terms[0:numTopicTerms]))
        return label

    def getGGplotCommands(self, outcome, file_in, file_out, featLabels=None, research=False):
        """
        Parameters
        -----------
        outcome : str
            name of the outcome to plot
        file_in : str
            input file name
        file_out : str
            output file name (plot is stored here)
        featLabels : :obj:`list`, optional
            list of feature labels
        research : :obj:`boolean`, optional
            False - Detailed color version of the plot
            True  - standard plot

        Returns
        -------
        commands : str
            Command string used to generate a plot

        """
        output_settings = "dpi=400, units='in', width=8, height=7"
        if not research:
            #GREY SCALE:
            # plotcommand = '''num_feats = length(factor(full.plotdata$feature));
            #   ggplot(data=full.plotdata, aes(x=%s, y=yhat, colour=feature)) +
            #   geom_line(aes(linetype=feature), size=3) +
            #   theme_bw(24) +
            #   scale_y_continuous('Standardized Relative Frequency') +
            #   scale_x_continuous('%s') +
            #   scale_colour_manual(name='Features', values=rep(gray.colors(3, 0, .8), ceiling(num_feats/3.0) ) ) +
            #   scale_linetype_manual(name="Features", values=1 + floor(0:(num_feats-1)/3)) +
            #   opts(legend.position="top",
            #        legend.direction="horizontal",
            #        legend.title=theme_blank(),
            #        legend.key=theme_rect(linetype='blank'),
            #        legend.text=theme_text(colour="black", size=15))'''%(outcome, outcome.capitalize())

            # plotcommand = '''num_feats = length(factor(full.plotdata$feature));
            #   ggplot(data=full.plotdata, aes(x=%s, y=yhat, colour=feature)) +
            #   geom_line(size=3) +
            #   theme_bw(24) +
            #   scale_y_continuous('Standardized Relative Frequency') +
            #   scale_x_continuous('%s') +
            #   scale_colour_manual(name='Features', values=c("red", "green", "blue", "black") ) +              
            #   opts(legend.position="top",
            #        legend.direction="horizontal",
            #        legend.title=theme_blank(),
            #        legend.key=theme_rect(linetype='blank'),
            #        legend.text=theme_text(colour="black", size=15))'''%(outcome, outcome.capitalize())
            ## ABOVE LAD

            plotcommand = '''num_feats = length(factor(full.plotdata$feature));
              ggplot(data=full.plotdata, aes(x=%s, y=yhat, colour=feature)) +
              geom_line(aes(linetype="solid"), size=2) +
              theme_bw(18) +
              scale_y_continuous('Standardized Relative Frequency') +
              scale_x_continuous('%s') +
              scale_colour_manual(name='Features', values=rep(rainbow(6), ceiling(9/3.0) ) ) +
              scale_linetype_manual(name="Features", values=1 + floor(0:(num_feats-1)/3)) +
              opts(legend.position="top",
                   legend.direction="horizontal",
                   legend.title=theme_blank(),
                   legend.key=theme_rect(linetype='blank'),
                   legend.text=theme_text(colour="black", size=18),
                   panel.grid.major = theme_blank(),
                   panel.grid.minor = theme_blank(),
                   panel.background = theme_blank()
            )'''%(outcome, outcome.capitalize(), )
            if outcome == 'age':
                plotcommand += ' + scale_x_continuous("%s", limits=c(13, 64), breaks=c(13,20,30,40,50,60))'%(outcome.capitalize(),)
            if featLabels:
                labelCommand = ', '.join(['"%s"'%(x,) for x in featLabels])
                plotcommand += ' + scale_colour_manual(name="Features", values=rep(rainbow(6), ceiling(9/3.0) ), labels=c(%s))'%(labelCommand,)
                plotcommand += ' + opts(legend.direction="vertical", legend.text=theme_text(colour="black", size=13))'
               
            commands = '''require(ggplot2);load("%s");%s;ggsave("%s", height=7, width=9, units="in");'''%(file_in, plotcommand, file_out)
            #commands = '''require(ggplot2);load("%s");%s;ggsave("%s");dev.off()'''%(file_in, plotcommand, file_out)
            print(commands)
        else:
            if outcome=='age':
                commands = '''require(ggplot2);load("%s");ggplot(data=full.plotdata, aes(x=%s, y=yhat, colour=feature)) + geom_line() + theme_bw() + scale_y_continuous('Standardized Relative Frequency') + scale_x_continuous('%s') + opts();ggsave("%s");dev.off()'''%(file_in, outcome, outcome.capitalize(), file_out)
            else: # standard commands
                commands = '''require(ggplot2);load("%s");ggplot(data=full.plotdata, aes(x=%s, y=yhat, colour=feature)) + geom_line() + theme_bw() + scale_y_continuous('Standardized Relative Frequency') + scale_x_continuous('%s') + opts();ggsave("%s");dev.off()'''%(file_in, outcome, outcome.capitalize(), file_out)

        # commands = map(lambda x:x + '\n', commands.split(';'))
        return commands    

    def wildcardMatch(self, string, list1):
        """
        Parameters
        ----------
        string : strstring
        list1 : list
            list of strings

        Returns
        -------
         : boolean
            True  - if string is in list1
            False - no match


        """
        for endI in range(3, len(string)+1):
            stringWild = string[0:endI]+'*'
            if stringWild in list1:
                return True
        return False


    def mapFeatureName(self, feat, mapping):
        """

        Parameters
        ----------
        feat : int
            feature id
        mapping : dict
            mapping dict for feature labels

        Returns
        -------
        newFeat : str
            new feature name


        """
        newFeat = feat
        try:
            newFeat = mapping[str(feat)]
        except KeyError:
            pass
        return newFeat

    def getLabelmapFromLexicon(self, lexicon_table): 
        """
        Returns a label map based on a lexicon. labelmap is {feat:concatenated_categories}

        Parameters
        ----------
        lexicon_table : str
            Lexicon table name

        Returns
        -------
        feat_to_label : dict
            a label map based on a lexicon. labelmap is {feat:concatenated_categories}

        """
        (conn, cur, curD) = mm.dbConnect(self.lexicondb, charset=self.encoding, use_unicode=self.use_unicode)
        sql = 'SELECT * FROM %s'%(lexicon_table)
        rows = mm.executeGetList(self.lexicondb, cur, sql, True, charset=self.encoding, use_unicode=self.use_unicode) #returns list of [id, feat, cat, ...] entries

        feat_to_label = {}
        for row in rows:
            feat = row[1].strip()
            if feat_to_label.get(feat, None):
                feat_to_label[ feat ] += "_%s"%row[2].strip()
            else:
                feat_to_label[ feat ] = row[2].strip()

        return feat_to_label

    def getLabelmapFromLabelmapTable(self, labelmap_table='', lda_id=None):
        """
        Parses a labelmap table and returns a python dictionary: {feat:feat_label}

        Parameters
        ----------
        labelmap_table : :obj:`str`, optional
            name of lable map table
        lda_id : :obj:`str`, optional
            lda model id

        Returns
        -------
        feat_to_label : dict
            {feat:feat_label}


        """
        if labelmap_table:
            pass
        elif lda_id:
            labelmap_table = "feat_to_label$%s"%lda_id
        elif not labelmap_table:
            raise Exception("must specify labelmap_table or lda_id")

        (conn, cur, curD) = mm.dbConnect(self.lexicondb, charset=self.encoding, use_unicode=self.use_unicode)
        sql = 'SELECT * FROM %s'%(labelmap_table)
        rows = mm.executeGetList(self.lexicondb, cur, sql, True, charset=self.encoding, use_unicode=self.use_unicode) #returns list of [feat, label, ...] entries

        feat_to_label = {}
        for row in rows:
            feat_to_label[ row[0] ] = row[1].strip()

        return feat_to_label

    def topicDupeFilterCorrels(self, correls, topicLex, maxWords = 15, filterThresh=0.25):
        """
        Filters out topics that have many similar words to those with a stronger correlation

        Parameters
        ----------
        correls : dict
            outcome:feature
        topicLex : str
            table name
        maxWords : :obj:`int`, optional
            max number of words
        filterThresh : :obj:`float`, optional
            filter threshold

        Returns
        -------
        newCorrels : dict
            filtered dict (same structure as correls)

        """
        #get topic information
        topicWords = self.getTopicWords(topicLex, maxWords)
        newCorrels = dict()

        for outcomeField, rs in sorted(correls.items()):
            sigRs = dict()
            posRs = [(k, v) for k, v in rs.items() if v[self.r_idx] > 0]
            negRs = [(k, (float(-1*v[self.r_idx]),) + v[1:]) for k, v in rs.items() if v[self.r_idx] < 0]

            keepList = self.getTopicKeepList(posRs, topicWords, filterThresh)
            keepList = keepList | self.getTopicKeepList(negRs, topicWords, filterThresh)
            
            #apply filter:
            newCorrels[outcomeField] = dict([item for item in rs.items() if item[0] in keepList])
            
        return newCorrels

    @staticmethod
    def getTopicKeepList(rs, topicWords, filterThresh=0.25):
        """
        Gets a list of topics to keep during duplication filtering

        Parameters
        ----------
        rs : list
        topicWords : list
            list of words
        filterThresh : :obj:`float`, optional
            Filter threshold


        Returns
        -------
        keptTopics : list
            list of topics to keep

        """
        rList = sorted(rs, key= lambda f: abs(f[1][0]) if not isnan(f[1][0]) else 0, reverse=True)
        usedWordSets = list() # a list of sets of topic words
        keptTopics = set()
        for (topic, rf) in rList:
            (r, p, n, ci, freq) = rf
            tw = topicWords.get(topic)
            if not tw: 
                fwc.warn("**The following topic had no words from the topic lexicion**")
                fwc.warn("[Topic Id: %s, R: %.3f, p: %.4f, N: %d, Freq: %d]\n" % (topic, r, p, n, freq))
                continue
            currentWords = set([t[0] for t in tw])
            shouldCont = False
            for otherWords in usedWordSets:
                if len(otherWords.intersection(currentWords)) > filterThresh*len(currentWords):
                    shouldCont = True
                    break
            usedWordSets.append(currentWords)
            if not shouldCont: 
                keptTopics.add(topic)
            if freq < 1000:
                fwc.warn("**The frequency for this topic was very small**")
                fwc.warn("[Topic Id: %s, R: %.3f, p: %.4f, N: %d, CI: (%.4f,%.4f), Freq: %d]\n" % (topic, r, p, n, ci[0], ci[1], freq))

        return keptTopics


    def printTagCloudData(self, correls, maxP = fwc.DEF_P, outputFile='', paramString = None, maxWords = 100, duplicateFilter = False, colorScheme='multi'):
        """
        Prints data that can be inputted into tag cloud software

        Parameters
        ----------
        correls : dict
            outcome: feature dict
        maxP : :obj:`float`, optional
            p-value max
        outputFile : :obj:`str`, optional
            output file name
        paramString : :obj:`str`, optional
            string to be printed to screen
        maxwords : :obj:`int`, optional
            max number of words
        duplicateFilter : :obj:`boolean`, optional
        colorScheme : :obj:`str`, optional
            argument for color scheme


        """
        if paramString: print(paramString + "\n")

        fsock = None
        if outputFile:
            print("outputting tagcloud to: %s" % (outputFile + '.txt'))
            fsock = open(outputFile + '.txt', 'w+')

        sys.stdout = fsock if outputFile else sys.__stdout__

        wordFreqs = None
        for outcomeField, rs in sorted(correls.items()):
            print("\n=========================\nTag Cloud Data for %s\n---------------------------" % outcomeField)
            #print 'Cutoff at p < %f ' % maxP
            sigRs = dict()
            # for k, v in sorted(rs.items(), key = lambda r: abs(r[1][0]) if not isnan(r[1][0]) else 0, reverse=True):
            for k, v in rs.items():
                if v[self.p_idx] < maxP:
                    sigRs[k] = v
                # else:
                #     break            
            posRs = [(k, v) for k, v in sigRs.items() if v[self.r_idx] > 0]
            negRs = [(k, (float(-1*v[self.r_idx]),) + v[1:]) for k, v in sigRs.items() if v[self.r_idx] < 0]
            if duplicateFilter:
                if not wordFreqs: 
                    wordFreqs = dict( [(w, v[self.freq_idx]) for (w, v) in list(rs.items()) if not ' ' in w] ) #word->freq dict
                if posRs: 
                    if len(posRs[0][1]) < self.correls_length - 1:
                        fwc.warn("printTagCloudData: not enough data or duplicateFilter option, skipping filter for %s posRs\n"%outcomeField)
                    else:
                        posRs = OutcomeAnalyzer.duplicateFilter(posRs, wordFreqs, maxWords * 3)
                if negRs: 
                    if len(negRs[0][1]) < self.correls_length - 1:
                        fwc.warn("printTagCloudData: not enough data or duplicateFilter option, skipping filter for %s negRs\n"%outcomeField)
                    else:
                        negRs = OutcomeAnalyzer.duplicateFilter(negRs, wordFreqs, maxWords * 3)

            print("PositiveRs:\n------------")
            if colorScheme == 'bluered':
                OutcomeAnalyzer.printTagCloudFromTuples(posRs, maxWords, colorScheme='blue', use_unicode=self.use_unicode)
            elif colorScheme == 'redblue':
                OutcomeAnalyzer.printTagCloudFromTuples(posRs, maxWords, colorScheme='red', use_unicode=self.use_unicode)
            else:
                OutcomeAnalyzer.printTagCloudFromTuples(posRs, maxWords, colorScheme=colorScheme, use_unicode=self.use_unicode)
            # OutcomeAnalyzer.plotWordcloudFromTuples(posRs, maxWords, outputFile + ".%s.%s"%(outcomeField, "posR"), wordcloud )
            
            print("\nNegative Rs:\n-------------")
            if colorScheme == 'bluered':
                OutcomeAnalyzer.printTagCloudFromTuples(negRs, maxWords, colorScheme='red', use_unicode=self.use_unicode)
            elif colorScheme == 'redblue':
                OutcomeAnalyzer.printTagCloudFromTuples(negRs, maxWords, colorScheme='blue', use_unicode=self.use_unicode)
            else:
                OutcomeAnalyzer.printTagCloudFromTuples(negRs, maxWords, colorScheme=colorScheme, use_unicode=self.use_unicode)
            # OutcomeAnalyzer.plotWordcloudFromTuples(negRs, maxWords, outputFile + ".%s.%s"%(outcomeField, "negR"), wordcloud )

        if outputFile:
            fsock.close()
            sys.stdout = sys.__stdout__

    def printTopicTagCloudData(self, correls, topicLex, maxP = fwc.DEF_P, paramString = None, maxWords = 15, maxTopics = 100, duplicateFilter=False, colorScheme='multi', outputFile='', useFeatTableFeats=False):
        """
        Prints Topic Tag Cloud data to text file

        Parameters
        ----------
        correls : dict
            outcome:feature
        topicLex : str
            table name
        maxP : :obj:`float`, optional
            p-value max
        paramString : :obj:`str`, optional
            string to be printed to screen
        maxWords : :obj:`int`, optional
            max number of words
        maxTopics : :obj:`int`, optional
            max number of topics
        duplicateFilter : :obj:`boolean`, optional
            use duplicate filter if True
        colorScheme : :obj:`str`, optional
            color scheme option for cloud
        outputFile : :obj:`str`, optional
            output file name
        useFeatTableFeats : :obj:`boolean`, optional


        """
        if paramString: print(paramString + "\n")

        fsock = None
        if outputFile:
            print("outputting topic tagcloud data to: %s" % (outputFile+'.txt'))
            fsock = open(outputFile+'.txt', 'w')

        sys.stdout = fsock if outputFile else sys.__stdout__
        
        #get topic information, in case the data words need to be used topicLex is a dictionary
        topicWords = self.getTopicWords(topicLex, 1000) if useFeatTableFeats else self.getTopicWords(topicLex, maxWords)
        
        wordFreqs = None
        if useFeatTableFeats:
            print('Using words from featuretable for visualization')
            sql = "SELECT feat, sum(value) FROM %s group by feat"%(self.getWordTable())
            # words = {word: int(freq), ...}
            wordFreqs = {t[0]: int(t[1]) for t in mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)}
            words = set(wordFreqs.keys())
            newTopicWords = dict()
            for cat, tw in topicWords.items():
                newTopicWords[cat] = [(term, weight) for term, weight in tw if term in words][:maxWords]
                # _warn('Topic %s: %d -> %d (%s)' % (cat,len(tw),len(newTopicWords[cat]),str(newTopicWords[cat][:5])))
            topicWords = newTopicWords
        

        # rs are the correlations for the categories
        for outcomeField, rs in sorted(correls.items()):
            print("\n==============================\nTopic Tag Cloud Data for %s\n--------------------------------" % outcomeField)
            sigRs = dict()
            for k, v in sorted(list(rs.items()), key = lambda r: abs(r[1][self.r_idx]) if not isnan(r[1][self.r_idx]) else 0, reverse=True):
                if v[self.p_idx] < maxP:
                    sigRs[k] = v
                else:
                    break            
            posRs = [(k, v) for k, v in sigRs.items() if v[self.r_idx] > 0]
            negRs = [(k, (float(-1*v[self.r_idx]),) + v[1:]) for k, v in sigRs.items() if v[self.r_idx] < 0]

            print("PositiveRs:\n------------")
            if colorScheme == 'bluered':
                OutcomeAnalyzer.printTopicListTagCloudFromTuples(posRs, topicWords, maxWords, maxTopics, duplicateFilter, wordFreqs = wordFreqs, colorScheme='blue', use_unicode=self.use_unicode)
            elif colorScheme == 'redblue':
                OutcomeAnalyzer.printTopicListTagCloudFromTuples(posRs, topicWords, maxWords, maxTopics, duplicateFilter, wordFreqs = wordFreqs, colorScheme='red', use_unicode=self.use_unicode)
    
            else:
                OutcomeAnalyzer.printTopicListTagCloudFromTuples(posRs, topicWords, maxWords, maxTopics, duplicateFilter, wordFreqs = wordFreqs, colorScheme=colorScheme, use_unicode=self.use_unicode)
            
            print("\nNegative Rs:\n-------------")
            if colorScheme == 'bluered':
                OutcomeAnalyzer.printTopicListTagCloudFromTuples(negRs, topicWords, maxWords, maxTopics, duplicateFilter, wordFreqs = wordFreqs, colorScheme='red', use_unicode=self.use_unicode)
            elif colorScheme == 'redblue':
                OutcomeAnalyzer.printTopicListTagCloudFromTuples(negRs, topicWords, maxWords, maxTopics, duplicateFilter, wordFreqs = wordFreqs, colorScheme='blue', use_unicode=self.use_unicode)            
            else:
                OutcomeAnalyzer.printTopicListTagCloudFromTuples(negRs, topicWords, maxWords, maxTopics, duplicateFilter, wordFreqs = wordFreqs, colorScheme=colorScheme, use_unicode=self.use_unicode)

        if outputFile:
            fsock.close()
            sys.stdout = sys.__stdout__


    def getTopicWords(self, topicLex, maxWords=15):
        """
        Gets the most prevalent words in a topic

        Parameters
        ----------
        topicLex : str
            lexicon table
        maxWords : :obj:`int`, optional
            max number of words to return

        Returns
        -------
        topicWords : dict
            {term : weight}
        """
        if not topicLex:
            fwc.warn("No topic lexicon selected, please specify it with --topic_lexicon TOP_LEX")
            exit(2)
        sql = "SELECT term, category, weight FROM %s.%s"%(self.lexicondb, topicLex)
        catList = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
        topicWords = dict()
        for (term, cat, w) in catList:
            stripped_cat = cat.strip() #thank you Daniel, love Johannes
            try:
                topicWords[stripped_cat].append((term, w))
            except KeyError:
                topicWords[stripped_cat] = [(term, w)]
        for stripped_cat, words in list(topicWords.items()):
           topicWords[stripped_cat] = sorted(words, key = lambda tw: tw[1], reverse = True)[:maxWords]
        #pprint(topicWords)
        return topicWords

    @staticmethod
    def printTopicListTagCloudFromTuples(rs, topicWords, maxWords = 25, maxTopics = 40, duplicateFilter = False, wordFreqs = None, filterThresh = 0.25, colorScheme='multi', use_unicode=True):
        """
        print topic tag cloud


        Parameters
        ----------
        rs : list
        topicWords : list
            list of words
        maxwords : :obj:`int`, optional
            max number of words
        maxTopics : :obj:`int`, optional
            max number of topics
        duplicateFilter : :obj:`boolean`, optional
            use duplicate filter if True
        wordFreqs : :obj:`list`, optional
            list of word frequencies
        filterThresh : :obj:`float`, optional
            Filter threshold
        colorScheme : :obj:`str`, optional
            color scheme option for cloud
        use_unicode : :obj:`boolean`, optional
            keep unicode characters is True

        Returns
        -------
        topicWords : list
            list of words
        """
        rList = sorted(rs, key= lambda f: abs(f[1][OutcomeAnalyzer.r_idx]) if not isnan(f[1][OutcomeAnalyzer.r_idx]) else 0, reverse=True)[:maxTopics]
        usedWordSets = list() # a list of sets of topic words
        for (topic, rf) in rList:
            print("\n")
            (r, p, n, ci, freq) = rf
            #print >> sys.stderr, topic
            # list of words per topic
            tw = topicWords.get(topic)
            if not tw: 
                print("**The following topic had no words from the topic lexicion**")
                print(("[Topic Id: %s, R: %.3f, p: %.4f, N: %d, CI: (%.4f, %.4f), Freq: %d]" % (topic, r, p, n, ci[0], ci[1], freq)).decode("utf-8"))
                continue
            if duplicateFilter:
                currentWords = set([t[0] for t in tw])
                shouldCont = False
                for otherWords in usedWordSets:
                    if len(otherWords.intersection(currentWords)) > filterThresh*len(currentWords):
                        shouldCont = True
                        break
                usedWordSets.append(currentWords)
                if shouldCont: 
                    #continue
                    print("**The following topic did not pass the duplicate filter**")
                    print("**[ %d matched out of %d total words ]**"%(len(otherWords.intersection(currentWords)) , len(currentWords)))
            if freq < 2000:
                print("**The frequency for this topic is too small**")

            print("[Topic Id: %s, R: %.3f, p: %.4f, N: %d, CI: (%.3f, %.3f), Freq: %d]" % (topic, r, p, n, ci[0], ci[1], freq))
            # if using 1gram words and frequencies
            # (r, p, n, freq) belong to the category only
            tw = [(w_f[0], (w_f[1], p, n, ci, w_f[1])) for w_f in tw] if not wordFreqs else [(w_f1[0], (w_f1[1]*float(wordFreqs[w_f1[0]]), p, n, ci, wordFreqs[w_f1[0]])) for w_f1 in tw]
            OutcomeAnalyzer.printTagCloudFromTuples(tw, maxWords, rankOrderR = True, colorScheme=colorScheme, use_unicode=use_unicode)

        # pprint(topicWords)
        return topicWords

    def buildTopicLabelDict(self, topic_lex, num_words=3):
        """
        Build a topic label dictionary

        Parameters
        ----------
        topic_lex : str
            name lex table
        num_words : :obj:`int`, optional
            number of words

        Returns
        -------
        topicLabels : list
            list of labels

        """
        topicLabels = {}
        topic_to_words = self.getTopicWords(topic_lex, num_words)
        invalid_char_replacements = {'\\':'_bsl_', '/':'_fsl_', ':':'_col_', '*':'_ast_', '?':'_ques_', '"':'_doubquo_', '<':'_lsthan_', '>':'_grthan_', '|':'_pipe_'}
        for topic in topic_to_words:
            words = [x[0] for x in topic_to_words[topic]]
            label = ' '.join(words)
            for char in invalid_char_replacements:
                label = label.replace(char, invalid_char_replacements[char])
            topicLabels[topic] = label
            # converts ["hi", "friend", ... into "hi friend ..."
            # topicLabels[cat] = ' '.join([words[x]+"\n" if (x+1) % ceil(sqrt(len(words))) == 0 else words[x] for x in xrange(len(words))]).replace("\n ", "\n")
            # converts ["hi", "friend", ... into "hi friend .. \nhello bingo"
        return topicLabels

    @staticmethod
    def plotWordcloudFromTuples(rList, maxWords, outputFile, wordcloud):
        """

        Parameters
        ----------
        rList : list
            list of (word, correl) tuples
        maxWords : int
            max number of words
        outputFile : str
            file name
        wordcloud : 

        """
        #rlist is a list of (word, correl) tuples
        if len(rList) < 3:
            print("rList has less than 3 items\n")
            return False
        rList = sorted(rList, key= lambda f: abs(f[1][OutcomeAnalyzer.r_idx]) if not isnan(f[1][OutcomeAnalyzer.r_idx]) else 0, reverse=True)[:maxWords]

        maxR = rList[0][1][OutcomeAnalyzer.r_idx]
        minR = rList[-1][1][OutcomeAnalyzer.r_idx]
        diff = float(maxR - minR)
        smallDataBump = max((maxWords - len(rList)), 15)
        #TODO: use freqs
        rList = [(w, int(((v[OutcomeAnalyzer.r_idx]-minR)/diff)*maxWords) + smallDataBump) for w, v in rList]

        (w, occ) = list(zip(*rList))
        #pickle.dump((rList, w, occ), open("DBG3", "wb"))
        wordcloud(w, occ, outputFile)

    @staticmethod
    def buildBatchPlotFile(corpdb, featTable, topicList=''):
        """
        Builds a file to be used for batch plotting


        Parameters
        ----------
        corpdb : str
            database name
        featTable : str
            feature table name
        topicList : :obj:`list`, optional
            list of strings containing topics

        Returns
        -------
        outputfile : str
            output file name

        """
        (conn, cur, curD) = mm.dbConnect(corpdb, charset=self.encoding, use_unicode=self.use_unicode)
        outputfile = '/tmp/flexiplot.csv'
        if topicList:
            from collections import OrderedDict
            outDict = OrderedDict([('title_name', ' '.join(topicList))] + [('topic_%d'%ii, topicList[ii]) for ii in range(len(topicList))])
            #pprint(outDict)
            csvOut = csv.DictWriter(open(outputfile, 'w'), fieldnames=list(outDict.keys()))
            csvOut.writerow(outDict)
        else:
            csvOut = csv.DictWriter(open(outputfile, 'w'), fieldnames=['title_name', 'feat'])
            sql = 'SELECT DISTINCT feat FROM %s'%featTable
            feats = mm.executeGetList1(corpdb, cur, sql, True, charset=self.encoding, use_unicode=self.use_unicode)
            for feat in feats:
                csvOut.writerow({'title_name':feat, 'feat':feat})
        return outputfile

    @staticmethod
    def plotFlexibinnedTable(corpdb, flexiTable, featureFile, feat_to_label=None, preserveBinTable=False):
        """
        Plots a flexi binned table

        Parameters
        ----------
        corpdb : str
            database name
        flexiTable : str
            table name
        featureFile : str
            feature file name
        feat_to_label : :obj:`list`, optional
            feature-label tuples
        preserveBinTable : :obj:`boolean`, optional
            True- preserve bin table, else drop table


        """
        (conn, cur, curD) = mm.dbConnect(corpdb, charset=self.encoding, use_unicode=self.use_unicode)
        (pconn, pcur, pcurD) = mm.dbConnect(self.lexicondb, charset=self.encoding, use_unicode=self.use_unicode)
        if not feat_to_label:
            sql = 'SELECT DISTINCT(feat) FROM %s'%flexiTable
            feats = mm.executeGetList(corpdb, cur, sql, charset=self.encoding, use_unicode=self.use_unicode)[0]
            feat_to_label = dict([(x,x) for x in feats])

        temp_table = 'feat_to_label_temp'
        biggest_feat = max(list(map(len, list(feat_to_label.keys()))))
        biggest_label = max(list(map(len, list(feat_to_label.values()))))
        sql = 'DROP TABLE IF EXISTS %s'%temp_table
        mm.execute(self.lexicondb, pcur, sql, charset=self.encoding, use_unicode=self.use_unicode)
        sql = 'CREATE TABLE %s (feat varchar(%d), label varchar(%d))'%(temp_table, biggest_feat, biggest_label)
        mm.execute(self.lexicondb, pcur, sql, charset=self.encoding, use_unicode=self.use_unicode)
        rows = []
        for feat, label in feat_to_label.items():
            rows.append((feat, label))
        sql = 'INSERT INTO %s VALUES(%s)'%(temp_table, '%s, %s')
        mm.executeWriteMany(self.lexicondb, pconn, sql, rows, writeCursor=self.dbConn.cursor(), charset=self.encoding)
        #pprint(mm.executeGetList(self.lexicondb, pcur, 'SELECT * FROM %s'%temp_table))
        cmd = "/usr/bin/Rscript plotbot.R 'from.file' '%s' '%s' 'feat_to_label_temp' '%s'"%(corpdb, flexiTable, featureFile)
        os.system(cmd)

        sql = 'DROP TABLE IF EXISTS feat_to_label_temp'
        mm.execute(self.lexicondb, pcur, sql, charset=self.encoding, use_unicode=self.use_unicode)

        # pprint(mm.executeGetList(corpdb, cur, 'SELECT group_id, N FROM %s GROUP BY group_id'%flexiTable))
        print((mm.executeGetList(corpdb, cur, 'SELECT group_id FROM %s GROUP BY group_id'%flexiTable, charset=self.encoding, use_unicode=self.use_unicode)))
        if not preserveBinTable:
            sql = 'DROP TABLE %s'%flexiTable
            mm.execute(corpdb, cur, sql, charset=self.encoding, use_unicode=self.use_unicode)

    @staticmethod
    def printTagCloudFromTuples(rList, maxWords, rankOrderFreq = True, rankOrderR = False, colorScheme='multi', use_unicode=True):
        """
        Prints a tag cloud from a set of tuples

        Parameters
        ----------
        rlist : list
            list of (word, correl) tuples
        maxWords : int
            maximum number of words for the cloud
        rankOrderFreq : :obj:`boolean`, optional
        rankOrderR : :obj:`boolean`, optional
        colorScheme : :obj:`str`, optional
            color scheme of plot
        use_unicode : :obj:`boolean`, optional
            When true include unicode in clouds


        """
        #rlist is a list of (word, correl) tuples
        if len(rList) < 1:
            print("rList has less than no items\n")
            return False
        rList = sorted(rList, key= lambda f: abs(f[1][OutcomeAnalyzer.r_idx]) if not isnan(f[1][OutcomeAnalyzer.r_idx]) else 0, reverse=True)[:maxWords]
        if rankOrderR:
            for i in range(len(rList)):
                newValues = rList[i]
                newValues = (newValues[0], ((maxWords - i),) + newValues[1][1:])
                rList[i] = newValues

        maxR = rList[0][1][OutcomeAnalyzer.r_idx]
        minR = rList[-1][1][OutcomeAnalyzer.r_idx]
        diff = float(maxR - minR)
        if diff == 0: diff = 0.000001
        smallDataBump = max((maxWords - len(rList)), 10)
        if rankOrderFreq and len(rList[0][1]) > OutcomeAnalyzer.correls_length - 1:
            sortedFreqs = sorted([v[OutcomeAnalyzer.freq_idx] for (w, v) in rList])
            rList = [(w, int(((v[OutcomeAnalyzer.r_idx]-minR)/diff)*maxWords) + smallDataBump, (1+sortedFreqs.index(v[OutcomeAnalyzer.freq_idx]))) for (w, v) in rList]
        else:
            rList = [(w, int(((v[OutcomeAnalyzer.r_idx]-minR)/diff)*maxWords) + smallDataBump, v[OutcomeAnalyzer.freq_idx] if len(v) > correls_length - 1 else None) for (w, v) in rList]

        maxFreq = 0
        if rList[0][2]:
            maxFreq = max([v[2] for v in rList])

        for (w, occ, freq) in rList:
            if freq:
                color = OutcomeAnalyzer.freqToColor(freq, maxFreq, colorScheme=colorScheme)
                if use_unicode:
                    print("%s:%d:%s" % (w.replace(' ', '_'), int(occ), color))
                else:
                    if len(w) > len(fwc.removeNonAscii(w)): 
                        fwc.warn("Unicode being ignored, %s is being skipped" % w)
                    else:
                        print("%s:%d:%s" % (w.replace(' ', '_'), int(occ), color))
                
            else:
                if use_unicode:
                    print("%s:%d" % (w.replace(' ', '_'), int(occ)))
                else:
                    if len(w) > len(fwc.removeNonAscii(w)): 
                        fwc.warn("Unicode being ignored, %s is being skipped" % w)
                    else:
                        print("%s:%d" % (w.replace(' ', '_'), int(occ)))
                

    @staticmethod
    def duplicateFilter(rList, wordFreqs, maxToCheck = 100):
        """
        Filters out duplicate words

        Parameters
        ----------
        rList : list
            list of (word, correl) tuples
        wordFreqs : dict
            word - word frequency pairs
        maxToCheck : :obj:`int`, optional
            will stop checking after this many in order to speed up operation

        Returns
        -------
        newList : list
            filtered version of rList

        """
        #maxToCheck, will stop checking after this many in order to speed up operation

        sortedList = sorted(rList, key= lambda f: abs(f[1][OutcomeAnalyzer.r_idx]) if not isnan(f[1][OutcomeAnalyzer.r_idx]) else 0, reverse=True)[:maxToCheck]
        usedWords = set()
        newList = []
        # pprint(('before filter', sortedList))#debug
        for (phrase, v) in sortedList:
            (r, sig, groups, ci, phraseFreq) = v
            words = phrase.split(' ')

            # check for keeping:
            keep = True
            for i in range(len(words)):
                word = words[i]
                if word in usedWords:
                    ditch = True
                    for otherWord in words[:i]+words[i+1:]:
                        if (not otherWord in wordFreqs) or (not word in wordFreqs) or wordFreqs[otherWord] < wordFreqs[word]:
                            ditch = False
                            break
                    keep = keep and not ditch
            if keep:
                # print "keeping %s" % phrase
                newList.append((phrase, v))
                for word in words:
                    usedWords.add(word)
            else:
                pass
                # print "ditching %s" % phrase

        # pprint(('after filter', newList))
        return newList


    @staticmethod
    def freqToColor(freq, maxFreq = 1000, resolution=64, colorScheme='multi'):
        """
        Alter color scheme of plot based on the the word frequencies

        Parameters
        ----------
        freq : int
            word frequency
        maxFreq : :obj:`int`, optional
            maximum frequency threshold
        resolution : :obj:`int`, optional
            pixels of resolution
        colorScheme : :obj:`str`, optional
            specifies color scheme of plot

        Returns
        -------
        htmlcode : str
            string of html code specifying color scheme


        """
        perc = freq / float(maxFreq)
        (red, green, blue) = (0, 0, 0)
        if colorScheme=='multi':
        #print "%d %d %.4f" %(freq, maxFreq, perc)#debug
            if perc < 0.17: #grey to darker grey
                (red, green, blue) = fwc.rgbColorMix((168, 168, 168),(124, 124, 148), resolution)[int(((1.00-(1-perc))/0.17)*resolution) - 1]
            elif perc >= 0.17 and perc < 0.52: #grey to blue
                (red, green, blue) = fwc.rgbColorMix((124, 124, 148), (32, 32, 210), resolution)[int(((0.830-(1-perc))/0.35)*resolution) - 1]
            elif perc >= 0.52 and perc < 0.90: #blue to red
                (red, green, blue) = fwc.rgbColorMix((32, 32, 210), (200, 16, 32), resolution)[int(((0.48-(1-perc))/0.38)*resolution) - 1]
            else: #red to dark red
                (red, green, blue) = fwc.rgbColorMix((200, 16, 32), (128, 0, 0), resolution)[int(((0.10-(1-perc))/0.10)*resolution) - 1]
        # blue:
        elif colorScheme=='blue':
            if perc < 0.50: #light blue to med. blue
                (red, green, blue) = fwc.rgbColorMix((76, 76, 236), (48, 48, 156), resolution)[int(((1.00-(1-perc))/0.5)*resolution) - 1]
            else: #med. blue to strong blue
                (red, green, blue) = fwc.rgbColorMix((48, 48, 156), (0, 0, 110), resolution)[int(((0.5-(1-perc))/0.5)*resolution) - 1]
        #red:
        elif colorScheme=='red': 
            if perc < 0.50: #light red to med. red
                (red, green, blue) = fwc.rgbColorMix((236, 76, 76), (156, 48, 48), resolution)[int(((1.00-(1-perc))/0.5)*resolution) - 1]
            else: #med. red to strong red
                (red, green, blue) = fwc.rgbColorMix((156, 48, 48), (110, 0, 0), resolution)[int(((0.5-(1-perc))/0.5)*resolution) - 1]
        elif colorScheme=='green': 
            (red, green, blue) = fwc.rgbColorMix((166, 247, 178), (27, 122, 26), resolution)[int((1.00-(1-perc))*resolution) - 1]
        #red+randomness:
        elif colorScheme=='red-random':
            if perc < 0.50: #light blue to med. blue
                (red, green, blue) = fwc.rgbColorMix((236, 76, 76), (156, 48, 48), resolution, True)[int(((1.00-(1-perc))/0.5)*resolution) - 1]
            else: #med. blue to strong blue
                (red, green, blue) = fwc.rgbColorMix((156, 48, 48), (110, 0, 0), resolution, True)[int(((0.5-(1-perc))/0.5)*resolution) - 1]


        #print "(%d %d %d)" %(red, green, blue)#debug

        htmlcode = "%02s%02s%02s" % (hex(red)[2:], hex(green)[2:], hex(blue)[2:])
        return htmlcode.replace(' ', '0')


    def generateTagCloudImage(self, correls, maxP = fwc.DEF_P, paramString = None, colorScheme='multi' ):
        """
        Generates a tag cloud image from correls

        Parameters
        ----------
        correls : dict
            outcome: feature dict
        maxP : :obj:`float`, optional
            p-value max
        paramString : :obj:`str`, optional
            string to be printed to screen
        colorScheme : :obj:`str`, optional
            argument for color scheme

        """
        if paramString: print(paramString + "\n")
        #TODO: make maxP a parameter
        maxWords = 150
        for outcomeField, rs in correls.items():
            sigRs = dict()
            for k, v in sorted(list(rs.items()), key = lambda r: abs(r[self.r_idx]) if not isnan(r[self.r_idx]) else 0, reverse=True):
                if v[self.p_idx] < maxP:
                    sigRs[k] = v[self.r_idx]
                else:
                    break            
            posRs = [(k, v) for k, v in sigRs.items() if v > 0]
            negRs = [(k, float(-1*v)) for k, v in sigRs.items() if v < 0]

            print("PositiveRs:\n------------")
            OutcomeAnalyzer.printTagCloudFromTuples(posRs, maxWords, colorScheme=colorScheme, use_unicode=self.use_unicode)
            print("\nNegative Rs:\n-------------")
            OutcomeAnalyzer.printTagCloudFromTuples(negRs, maxWords, colorScheme=colorScheme, use_unicode=self.use_unicode)

    @staticmethod
    def generateTagCloudImageFromTuples(rList, maxWords):
        """
        Generates a tag cloud from a list of tuples


        Parameters
        ----------
        rList : list
        maxWords : int

        """
        #rlist is a list of (word, correl) tuples
        if len(rList) < 3:
            print("rList has less than 3 items\n")
            return False
        rList = sorted(rList, key= lambda f: abs(f[self.r_idx]) if not isnan(f[self.r_idx]) else 0, reverse=True)[:maxWords]

        maxR = rList[0][1]
        minR = rList[-1][1]
        diff = float(maxR - minR)
        smallDataBump = max((maxWords - len(rList)), 10)
        rList = [(w, int(((r-minR)/diff)*maxWords) + smallDataBump) for (w, r) in rList]

        #uncomment to print raw word occurences
        #for (w, occ) in rList:
        #    print str(w+' ')*occ
        #print ' '
        for (w, occ) in rList:
            print("%s:%d" % (str(w), int(occ)))

                              
    def barPlot(self, correls, outputFile = None, featSet = set(), featsPerOutcome = 5):
        from rFeaturePlot import FeaturePlotter        
        fwc.warn("Generating Bar Plot.")
        if not outputFile: outputFile = '_'.join(self.outcome_value_fields)
        #generate features to use: (ignore featsPerOutcome if featset is passed in)
        if not featSet:
            for outcomeField, featRs in correls.items():
                featSet.update(sorted(list(featRs.keys()), key = lambda k: 
                                      featRs[k] if not isnan(featRs[k][0]) else 0 , 
                                      reverse = True)[:featsPerOutcome])
        newCorrels = dict()
        for feat in featSet:
            newCorrels[feat] = dict()
            for outcomeField, featRs in correls.items():
                newCorrels[feat][outcomeField] = featRs[feat]
        FeaturePlotter().barPlot(outputFile, newCorrels) 
                               
    def correlMatrix(self, correlMatrix, outputFile = None, outputFormat='html', sort = False, pValue = True, nValue = True, cInt = True, freq = False, paramString = None):
        fwc.warn("Generating Correlation Matrix.")
                    
        if outputFile: 
            #redirect
            outputFile = outputFile +'.'+outputFormat
            fwc.warn(" print to file: %s" % outputFile)
        outputFilePtr = sys.stdout
        if outputFile: outputFilePtr = open(outputFile, 'w') 
        if paramString: print(paramString + "\n \n", file=outputFilePtr)
            
        outputFormat = outputFormat.lower()
        fwc.warn('=====================%d===================='%(len(correlMatrix),))
        if outputFormat == 'pickle':
            import pickle as pickle
            pickle.dump(correlMatrix, open(outputFile, "wb" ))
        elif outputFormat == 'csv':
            if (len(correlMatrix)>0): 
                self.outputCorrelMatrixCSV(fwc.reverseDictDict(correlMatrix), pValue, nValue, cInt, freq, outputFilePtr=outputFilePtr)
                if sort: self.outputSortedCorrelCSV(correlMatrix, pValue, nValue, cInt, freq, outputFilePtr=outputFilePtr)
        elif outputFormat == 'html':
            if (len(correlMatrix)>0):
                header = '<head><meta charset="UTF-8"></head><br><br><div id="top"><a href="#feats">Features by alphabetical order</a>'
                if sort: header += ' or see them <a href="#sorted">sorted by r-values</a>'
                header += '<br><br></div>'
                print(header, file=outputFilePtr)
                self.outputCorrelMatrixHTML(fwc.reverseDictDict(correlMatrix), pValue, nValue, cInt, freq, outputFilePtr=outputFilePtr)
                if sort: self.outputSortedCorrelHTML(correlMatrix, pValue, nValue, cInt, freq, outputFilePtr=outputFilePtr)
        else:
            fwc.warn("unknown output format: %s"% outputFormat)
      
    @staticmethod    
    def outputCorrelMatrixCSV(correlMatrix, pValue = True, nValue = True, cInt=True, freq=True, outputFilePtr = sys.stdout):
        keys1 = sorted(list(correlMatrix.keys()), key = fwc.permaSortedKey)
        # keys2 = sorted(correlMatrix[keys1[0]].keys(), key = permaSortedKey)
        keys2 = set([j for i in list(correlMatrix.values()) for j in list(i.keys())])
        keys2 = sorted(list(keys2), key = fwc.permaSortedKey)
        writer = csv.writer(outputFilePtr)
        titlerow = ['feature']
        for key2 in keys2:
            titlerow.append(fwc.tupleToStr(key2))
            if pValue: titlerow.append('p')
            if nValue: titlerow.append('N')
            if cInt:
                titlerow.append('CI_l')
                titlerow.append('CI_u')
            if freq: titlerow.append('freq')
        writer.writerow(titlerow)
        for key1 in keys1:
            row = [fwc.tupleToStr(key1)]
            for key2 in keys2:
                (r, p, n, ci, f) = correlMatrix[key1].get(key2, [0, 1, 0, (0,0), 0])[:OutcomeAnalyzer.correls_length]
                row.append(r)
                if pValue: row.append(p)
                if nValue: row.append(n)
                if cInt:
                    row.append(ci[0])
                    row.append(ci[1])
                if freq: row.append(f)
            try:
                writer.writerow(row)
            except UnicodeEncodeError:
                fwc.warn("Line contains unprintable unicode, skipped: %s" % row)

    @staticmethod
    def outputSortedCorrelCSV(correlMatrix, pValue = True, nValue = True, cInt = True, freq=False, outputFilePtr = sys.stdout, topN=50):
        """Ouputs a sorted correlation matrix (note correlmatrix is reversed from non-sorted) """
        #TODO: topN
        print("\nSORTED:", file=outputFilePtr)
        keys1 = sorted(list(correlMatrix.keys()), key = fwc.permaSortedKey)
        sortedData = dict()
        titlerow = ['rank']
        maxLen = 0
        for key1 in keys1:
            sortedData[key1] = sorted(list(correlMatrix[key1].items()), key=lambda k_v: (-1*float(k_v[1][OutcomeAnalyzer.r_idx]) if not isnan(float(k_v[1][OutcomeAnalyzer.r_idx])) else 0, k_v[0]))
            if len(sortedData[key1]) > maxLen: maxLen = len(sortedData[key1])
            titlerow.append(key1)
            titlerow.append('r')
            if pValue: titlerow.append('p')
            if nValue: titlerow.append('N')
            if cInt: 
                titlerow.append('CI_l')
                titlerow.append('CI_u')
            if freq: titlerow.append('freq')
        writer = csv.writer(outputFilePtr)
        writer.writerow(titlerow)

        for rank in range(maxLen):
            row = [int(rank+1)]
            for key1 in keys1:
                data = sortedData[key1]
                if rank < len(data):#print this keys rank item
                    data = data[rank]
                    row.append(fwc.tupleToStr(data[0])) #name of feature
                    (r, p, n, ci, f) = data[1][:OutcomeAnalyzer.correls_length]
                    row.append(r)
                    if pValue: row.append(p)
                    if nValue: row.append(n)
                    if cInt:
                        row.append(ci[0])
                        row.append(ci[1])
                    if freq: row.append(f)
            try:
                writer.writerow(row)
            except UnicodeEncodeError:
                fwc.warn("Line contains unprintable unicode, skipped: %s" % row)

        titlerow = ['rank']            
        for key1 in keys1:
            titlerow.append(key1)
            titlerow.append('r')
            if pValue: titlerow.append('p')
            if nValue: titlerow.append('N')
            if cInt:
                titlerow.append('CI_l')
                titlerow.append('CI_u')
            if freq: titlerow.append('freq')
        writer = csv.writer(outputFilePtr)
        writer.writerow(titlerow)



    @staticmethod    
    def outputComboCorrelMatrixCSV(comboCorrelMatrix, outputstream = sys.stdout, paramString = None):
        """prints correl matrices for all combinations of features, always prints p-values, n, and freq)"""
        print(paramString+"\n", file=outputstream)
        outcomeKeys = sorted(list(comboCorrelMatrix.keys()), key = fwc.permaSortedKey)
        for outcomeName in outcomeKeys:
            columnKeys = sorted(list(comboCorrelMatrix[outcomeName].keys()), key = lambda k: len(k))
            columnNames = ['feature'] + sum([['b_'+str(ck), 'p_'+str(ck), 'n_'+str(ck)] for ck in columnKeys], [])[:] + ['freq']
            print("\n"+outcomeName+"\n", file=outputstream)
            csvOut = csv.DictWriter(outputstream, fieldnames=columnNames)
            firstRow = dict([(str(k), str(k)) for k in columnNames])
            csvOut.writerow(firstRow)
            rowKeys = sorted(comboCorrelMatrix[outcomeName][columnKeys[-1]].keys()) #uses the last to capture all possible controls
            for rk in rowKeys:
                rowDict = {'feature': rk}
                for ck in columnKeys:
                    tup = ()
                    try: 
                        tup = comboCorrelMatrix[outcomeName][ck][rk]
                    except KeyError: #row doesn't have keep: skip
                        tup = (None, None, None)
                    if len(tup) > 3:
                        (rowDict['b_'+str(ck)], rowDict['p_'+str(ck)], rowDict['n_'+str(ck)], rowDict['freq']) = tup[:4]
                    else:
                        (rowDict['b_'+str(ck)], rowDict['p_'+str(ck)], rowDict['n_'+str(ck)]) = tup[:3]
                csvOut.writerow(rowDict)



    #HTML OUTPUTS:
    @staticmethod
    def outputCorrelMatrixHTML(correlMatrix, pValue = True, nValue = True, cInt = True, freq=False, outputFilePtr = sys.stdout):
        output="""<style media="screen" type="text/css">
                     table, th, td {border: 1px solid black;padding:2px;}
                     table {border-collapse:collapse;font:10pt verdana,arial,sans-serif;}
                     .sml {font:9pt;}
                     .tny {font:7pt;}
                     .fgsupersig {color:rgb(0, 0, 0);}
                     .fgsig {color:rgb(48, 48, 48);}
                     .fgunsig {color:rgb(100, 100, 100);}
                     .fgsuperunsig {color:rgb(150, 150, 150);}
                  </style>"""
        output += '<div id="feats">'
        output += "<table border=1><tr>"
        output += '<td></td>'
        # correlMatrix: dictionary of rows - dictionary of columns
        keys1 = sorted(list(correlMatrix.keys()), key = fwc.permaSortedKey)
        # keys2 = sorted(correlMatrix[keys1[0]].keys(), key = permaSortedKey)
        keys2 = set([j for i in list(correlMatrix.values()) for j in list(i.keys())])
        keys2 = sorted(list(keys2), key = fwc.permaSortedKey)
        lastKey2 = keys2[-1]
        for key2 in keys2:

            output += "<td><b>%s</b></td>" % fwc.tupleToStr(key2)
            if pValue: output += '<td class="sml"><em>(p)</em></td>'
            if nValue: output += '<td class="sml"><em>N</em></td>'
            if cInt: 
                output += '<td class="sml"><em>CI_l</em></td>'
                output += '<td class="sml"><em>CI_u</em></td>'
            if freq and (key2 == lastKey2): output += '<td class="tny">freq</td>'
        output += "</tr>\n"

        for key1 in keys1:
            output += "<tr><td><b>%s</b></td>" % fwc.tupleToStr(key1)
            ffreq = 0
            for key2 in keys2:
                (r, p, n, ci, f) = correlMatrix[key1].get(key2, [0, 1, 0, (0,0), ffreq])[:OutcomeAnalyzer.correls_length]
                print(r, p, n, ci, f)
                if not f: f = 0
                if f: ffreq = f
                
                #Add colors based on values
                fgclass = "fgsuperunsig"
                r_color = 0;
                if (p >= 0) and (r >= 0 or r < 0):
                    if p <= 0.01:
                        fgclass = "fgsupersig"
                    elif p <= 0.05:
                        fgclass = "fgsig"
                    elif p <= 0.1:
                        fgclass = "fgunsig" 
                    r_color = int(max(255 - floor(fabs(r)*400),0))
                bgcolor = ''
                if (r > 0):
                    bgcolor = "rgb(%s, 255, %s)" % (r_color, r_color)
                elif (r <= 0): #check for nan
                    bgcolor = "rgb(255, %s, %s)" % (r_color, r_color)
                else: #must be nan or inf
                    bgcolor = "rgb(190, 190, 190)"

                output += "<td class='%s' style='background-color:%s;'>%6.3f</td>" % (fgclass, bgcolor, float(r))
                if pValue:
                    output += "<td class='%s' style='background-color:%s;font-size:xx-small;border-left:0px;'><em>(%6.4f)</em></td>" % (fgclass, bgcolor, float(p))
                if nValue:
                    output += "<td class='%s' style='background-color:%s;font-size:xx-small;border-left:0px;'><em>%s</em></td>" % (fgclass, bgcolor, str(n))
                if cInt:
                    output += "<td class='%s' style='background-color:%s;font-size:xx-small;border-left:0px;'><em>%6.4f</em></td>" % (fgclass, bgcolor, float(ci[0]))
                    output += "<td class='%s' style='background-color:%s;font-size:xx-small;border-left:0px;'><em>%6.4f</em></td>" % (fgclass, bgcolor, float(ci[1]))
                if freq and (key2 == lastKey2):
                    output += "<td class='tny'>%s</td>" % f

            output += "</tr>\n"
        output += "</tr></table>\n"
        output += '<a href="#top">Back to top</a>'
        output += '</div>'
        print(output, file=outputFilePtr)

    @staticmethod
    def outputSortedCorrelHTML(correlMatrix, pValue = True, nValue = True, cInt = True, freq=False, outputFilePtr = sys.stdout):
        output="""<style media="screen" type="text/css">
                     table, th, td {border: 1px solid black;padding:2px;}
                     table {border-collapse:collapse;font:10pt verdana,arial,sans-serif;}
                     .sml {font:9pt;}
                     .tny {font:7pt;}
                     .fgsupersig {color:rgb(0, 0, 0);}
                     .fgsig {color:rgb(48, 48, 48);}
                     .fgunsig {color:rgb(100, 100, 100);}
                     .fgsuperunsig {color:rgb(150, 150, 150);}
                  </style>"""
        output += '<div id="sorted">'
        output += '<a href="#bottomSorted">Go to bottom</a>'
        output += "<p> SORTED:"
        output += "<table border=1>"
        keys1 = sorted(list(correlMatrix.keys()), key = fwc.permaSortedKey)
        sortedData = dict()
        maxLen = 0
        output += "<tr><td><b>rank</b></td>"
        for key1 in keys1:
            sortedData[key1] = sorted(iter(correlMatrix[key1].items()), key=lambda k_v2: (-1*float(k_v2[1][OutcomeAnalyzer.r_idx]) if not isnan(float(k_v2[1][OutcomeAnalyzer.r_idx])) else 0, k_v2[0]))
            if len(sortedData[key1]) > maxLen: maxLen = len(sortedData[key1])
            output += "<td><b>%s<br/>r</b></td>"%fwc.tupleToStr(key1)

            pnf = []
            if pValue: pnf.append("<em>(p)</em>")
            if nValue: pnf.append("<em>N</em>")
            if cInt:
                pnf.append("<em>CI_l</em>")
                pnf.append("<em>CI_u</em>")
            if freq: pnf.append("freq")
            if pnf:
                if len(pnf) > 1:
                    output += "<td class='tny'>"
                else:
                    output += "<td class='sml'>"
                output += "<br />".join(pnf)
                output += "</td>"

        output += "</tr>\n"

        for rank in range(maxLen):
            output += "<tr><td>%d</td>" % rank
            for key1 in keys1:
                data = sortedData[key1]
                if rank < len(data):#print this keys rank item
                    data = data[rank]
                    (r, p, n, ci, f) = data[1][:OutcomeAnalyzer.correls_length]

                    #Add colors based on values
                    fgclass = "fgsuperunsig"
                    r_color = 0;
                    if (p >= 0) and (r >= 0 or r < 0):
                        if p <= 0.01:
                            fgclass = "fgsupersig"
                        elif p <= 0.05:
                            fgclass = "fgsig"
                        elif p <= 0.1:
                            fgclass = "fgunsig" 
                        r_color = int(max(255 - floor(fabs(r)*400),0))
                    bgcolor = ''
                    if (r > 0):
                        bgcolor = "rgb(%s, 255, %s)" % (r_color, r_color)
                    elif (r <= 0): #check for nan
                        bgcolor = "rgb(255, %s, %s)" % (r_color, r_color)
                    else: #must be nan or inf
                        bgcolor = "rgb(190, 190, 190)"

                    output += "<td class='%s' style='background-color:%s;'>%s<br />%.3f</td>" % (fgclass, bgcolor,fwc.tupleToStr(data[0]), r)
                    pnf = []
                    if pValue: pnf.append("<em>(%6.4f)</em>"%float(p))
                    if nValue: pnf.append("<em>%d</em>" % int(n))
                    if cInt:
                        pnf.append("<em>%6.4f</em>" % float(ci[0]))
                        pnf.append("<em>%6.4f</em>" % float(ci[1]))
                    if freq: pnf.append(str(f))
                    if pnf:
                        output += "<td class='%s' style='background-color:%s;font-size:xx-small;border-left:0px;'>" %(fgclass, bgcolor)
                        output += "<br />".join(pnf)
                        output += "</td>"
            output += "</tr>\n"

        output += "<tr><td><b>rank</b></td>"
        for key1 in keys1:
            output += "<td><b>%s<br/>r</b></td>"%fwc.tupleToStr(key1)
            pnf = []
            if pValue: pnf.append("<em>(p)</em>")
            if nValue: pnf.append("<em>N</em>")
            if cInt:
                pnf.append("<em>CI_l</em>")
                pnf.append("<em>CI_u</em>")
            if freq: pnf.append("freq")
            if pnf:
                if len(pnf) > 1:
                    output += "<td class='tny'>"
                else:
                    output += "<td class='sml'>"
                output += "<br />".join(pnf)
                output += "</td>"
        output += "</tr>\n"
        output += "</table>\n"
        output += '<div id="bottomSorted"><a href="#top">Back to top</a> or go <a href="#sorted">back to top of sorted features</a></div>'
        output += '</div>'
        print(output, file=outputFilePtr)
        print("</p>", file=outputFilePtr)

    def printSignificantCoeffs(self, coeffs, outputFile = None, outputFormat='tsv', sort = False, pValue = True, nValue = False, maxP = fwc.DEF_P, paramString = None):
        fwc.warn("Generating Significant Coeffs.")
                               
        if outputFile: 
            #redirect
            outputFile = outputFile +'.'+outputFormat
            fwc.warn(" print to file: %s" % outputFile)
            old_stdout = sys.stdout
            sys.stdout = open(outputFile, 'w')
        if paramString: print(paramString + "\n")
            
        #loop through and find sig. correls (TODO: make a oneliner?)
        sigCoeffs = dict()
        for feat, outcomeCoeffs in coeffs.items():
            keep = False
            for outcomeName, values in outcomeCoeffs.items():
                if values[self.p_idx] <= maxP:
                    keep = True
                    break
            if keep:
                sigCoeffs[feat] = outcomeCoeffs #add all coeffs if any are sig
                
        #print output:
        outputFormat = outputFormat.lower()
        pprint(sigCoeffs)
        if outputFormat == 'html':
            raise NotImplementedError
        else:
            fwc.warn("unknown output format: %s"% outputFormat)
      
        if outputFile: 
            #unredirect
            sys.stdout = old_stdout

    def writeSignificantCoeffs4dVis(self, coeffs, outputFile, outputFormat='tsv', sort = False, pValue = True, nValue = False, maxP = fwc.DEF_P, paramString = None, interactions = False):
        fwc.warn("Generating Significant Coeffs.")

        # import pdb
        # pdb.set_trace()
        if paramString: print(paramString + "\n")
            
        #loop through and find sig. correls (TODO: make a oneliner?)
        sigCoeffs = dict()
        for feat, outcomeCoeffs in coeffs.items():
            keep = False
            for outcomeName, values in outcomeCoeffs.items():
                if values[self.p_idx] <= maxP:
                    keep = True
                    break
            if keep:
                sigCoeffs[feat] = outcomeCoeffs #add all coeffs if any are sig

        outcomes = list(sigCoeffs[list(sigCoeffs.keys())[0]].keys())

        header = '\t'.join(['ngram'] + [x + '-r' for x in outcomes] + [x + '-pValue' for x in outcomes] + ['n', 'freq']) + '\n'

        if outputFormat == 'tsv':
            outputFile += '.tsv.gz'
            with gzip.open(outputFile, 'wb') as gf:
                gf.write(header)
                print(header)
                for ngram, outcomeDict in sigCoeffs.items():
                    if '\t' in ngram:
                        fwc.warn('ngram [%s] skipped because it had a tab in it' % (ngram, ))
                        continue
                    # linedata has: ngram, outcome_1_r, outcome_2_r, ... outcome_n_r, outcome_1_pValue, outcome_2_pValue, ... outcome_n_pValue, num_users, ngram_frequency
                    linedata = [ngram] + list([outcomeDict[x][self.r_idx] for x in outcomes]) + list([outcomeDict[x][self.p_idx] for x in outcomes]) + list(outcomeDict[outcomes[0]][self.n_idx:4])
                    line = '\t'.join(map(str, linedata)) + '\n'
                    gf.write(line)
                    print(line)
        else:
            fwc.warn("unknown output format: %s"% outputFormat)
        pprint(sigCoeffs)

    def tableToDenseCsv(self, row_column, col_column, value_column, output_csv_filename=None, compress_csv=True):
        """Take a mysql table to convert a long table (e.g. feature table) to a dense 2x2 contingency matrix (size N by 
        M where N is the number of distinct rows and M is the number of distinct columns). Efficient (uses lookups instead 
        of a single iteration through all entries of the contingency matrix -- could be more slightly more efficient if it 
        used the dbCursor pointer).
        
        Parameters
        ----------
        row_column : str
            table column that will populate the rows of the contingency csv
        col_column : str
            table column that will populate the columns of the contingency csv
        value_column : str
            table column that will populate the values at the intersection of the rows and columns of the contingency csv
        output_csv_filename : str
            the name of the output file -- if empty is created based on the values provided
        compress_csv : boolean
            whether to gzip the csv
        """

        if not output_csv_filename:
            output_csv_filename = 'dense.{db}.{table}.{row}-by-{col}.{value}.csv'.format(db=self.corpdb, table=self.corptable, row=row_column, col=col_column, value=value_column)

        row_sql = 'SELECT DISTINCT {row} FROM {table} ORDER BY {row}'.format(row=row_column, table=self.corptable)
        sorted_row_values = list(mm.executeGetList1(self.corpdb, self.dbCursor, row_sql, charset=self.encoding, use_unicode=self.use_unicode))
        col_sql = 'SELECT DISTINCT {col} FROM {table} ORDER BY {col}'.format(col=col_column, table=self.corptable)
        sorted_col_values = list(mm.executeGetList1(self.corpdb, self.dbCursor, col_sql, charset=self.encoding, use_unicode=self.use_unicode))
        value_sql = 'SELECT {row}, {col}, {value} FROM {table} ORDER BY {row}, {col}'.format(row=row_column, col=col_column, value=value_column, table=self.corptable)
        sorted_values = mm.executeGetList(self.corpdb, self.dbCursor, value_sql, charset=self.encoding, use_unicode=self.use_unicode)

        N = len(sorted_row_values)
        M = len(sorted_col_values)

        with open(output_csv_filename, 'w') as output_csv:
            csv_writer = csv.writer(output_csv)
            csv_writer.writerow([row_column] + sorted_col_values)

            current_row = sorted_values[0][0]
            current_column_data = ['NULL'] * M
            num_row_writes = 0

            for tablerow, tablecol, tablevalue in sorted_values:
                # if a new row, write our current column data
                # and reset local variables
                if current_row != tablerow:
                    csv_writer.writerow([current_row] + current_column_data)
                    current_column_data = ['NULL'] * M
                    num_row_writes += 1
                    current_row = tablerow
                    if num_row_writes % 1000 == 0:
                        print('{n} out of {N} rows complete'.format(n=num_row_writes, N=N))
                column_index = sorted_col_values.index(tablecol)
                current_column_data[column_index] = tablevalue

            csv_writer.writerow([current_row] + current_column_data)
            print("wrote %d features over %d rows" % (N, num_row_writes))


        if compress_csv:
            # os.system("gzip {output_filename}".format(output_filename=output_csv_filename)
            pass