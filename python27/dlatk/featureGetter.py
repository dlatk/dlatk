from ConfigParser import SafeConfigParser
import MySQLdb
import pandas as pd

#math / stats:
from numpy import zeros, sqrt, array, std, mean
from scipy.stats import t as spt
import numpy as np

#infrastructure
import fwConstants as fwc
from featureWorker import FeatureWorker
from mysqlMethods import mysqlMethods as mm
from mysqlMethods import mysql_iter_funcs as mif

class FeatureGetter(FeatureWorker):
    """ General class for feature selection

        Attributes:
            corpdb:
            corptable:
            correl_field:
            mysql_host:
            message_field:
            messageid_field:
            encoding:
            use_unicode:
            lexicondb:
            featureTable:
            featNames:
            wordTable:

    """

    @classmethod
    def fromFile(cls, initFile):
        
        """
        Loads specified features from file

        Args: 
            initFile (string): path to file
            
        Example:
            creates a FeatureGetter Object with the features
            specified in the initFile
            FeatureGetter.fromFile('~/myInit.ini')
        """

        parser = SafeConfigParser()
        parser.read(initFile)
        corpdb = parser.get('constants','corpdb') if parser.has_option('constants','corpdb') else fwc.DEF_CORPDB
        corptable = parser.get('constants','corptable') if parser.has_option('constants','corptable') else fwc.DEF_CORPTABLE
        correl_field = parser.get('constants','correl_field') if parser.has_option('constants','correl_field') else fwc.DEF_CORREL_FIELD
        mysql_host = parser.get('constants','mysql_host') if parser.has_option('constants','mysql_host') else "localhost"
        message_field = parser.get('constants','message_field') if parser.has_option('constants','message_field') else fwc.DEF_MESSAGE_FIELD
        messageid_field = parser.get('constants','messageid_field') if parser.has_option('constants','messageid_field') else fwc.DEF_MESSAGEID_FIELD
        encoding = parser.get('constants','encoding') if parser.has_option('constants','encoding') else fwc.DEF_ENCODING
        if parser.has_option('constants','use_unicode'):
            use_unicode = True if parser.get('constants','use_unicode')=="True" else False
        else:
            use_unicode = fwc.DEF_UNICODE_SWITCH
        lexicondb = parser.get('constants','lexicondb') if parser.has_option('constants','lexicondb') else fwc.DEF_LEXICON_DB
        featureTable = parser.get('constants','feattable') if parser.has_option('constants','feattable') else fwc.DEF_FEAT_TABLE
        featNames = parser.get('constants','featnames') if parser.has_option('constants','featnames') else fwc.DEF_FEAT_NAMES
        wordTable = parser.get('constants','wordTable') if parser.has_option('constants','wordTable') else None
        return cls(corpdb=corpdb, corptable=corptable, correl_field=correl_field, mysql_host=mysql_host, message_field=message_field, messageid_field=messageid_field, encoding=encoding, use_unicode=use_unicode, lexicondb=lexicondb, featureTable=featureTable, featNames=featNames, wordTable = None)


    def __init__(self, corpdb=fwc.DEF_CORPDB, corptable=fwc.DEF_CORPTABLE, correl_field=fwc.DEF_CORREL_FIELD, mysql_host="localhost", message_field=fwc.DEF_MESSAGE_FIELD, messageid_field=fwc.DEF_MESSAGEID_FIELD, encoding=fwc.DEF_ENCODING, use_unicode=fwc.DEF_UNICODE_SWITCH, lexicondb = fwc.DEF_LEXICON_DB, featureTable=fwc.DEF_FEAT_TABLE, featNames=fwc.DEF_FEAT_NAMES, wordTable = None):
        super(FeatureGetter, self).__init__(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, encoding, use_unicode, lexicondb, wordTable=wordTable)
        self.featureTable = featureTable    
        self.featNames = featNames

    ##MAINTENANCE OPERATIONS##

    def optimizeFeatTable(self):
        """Optimizes the table -- good after a lot of deletes"""
        return mm.optimizeTable(self.corpdb, self.dbCursor, self.featureTable, charset=self.encoding, use_unicode=self.use_unicode)

    def disableFeatTableKeys(self):
        """Disable keys: good before doing a lot of inserts"""
        return mm.disableTableKeys(self.corpdb, self.dbCursor, self.featureTable, charset=self.encoding, use_unicode=self.use_unicode)

    def enableFeatTableKeys(self, table = None):
        """
        Enables the keys, for use after inserting (and with keys disabled)
        
        Args: 
            table: ????????????????????????????

        Example: 

        """
        return mm.enableTableKeys(self.corpdb, self.dbCursor, self.featureTable, charset=self.encoding, use_unicode=self.use_unicode)

    ## Getters ##

    def getFeatureCounts(self, groupFreqThresh = 0, where = '', SS = False, groups = set()):
        """
        Gets feature occurence by group
        
        Args: 
            groupFreqThresh (int): Minimum number of words a group must contain to be considered valid
            where (string): Conditional sql string to limit the search to elements meeting a specified criteria
            SS (boolean): Indicates the use of SSCursor (true use SSCursor to access MySQL)
            groups (set): Set of group ID's
        Returns:
            returns a list of (feature, count) tuples, 
            where count is the feature occurence in each group

        """

        if groupFreqThresh:
            groupCnts = self.getGroupWordCounts(where)
            for group, wordCount in groupCnts.iteritems():
                if (wordCount >= groupFreqThresh):
                    groups.add(group)
                    
        if (where): 
            where += ' WHERE ' + where
            if groups:
                where += ' AND ' + " group_id in ('%s')" % "','".join(str(g) for g in groups)
        elif groups:
            where = " WHERE group_id in ('%s')" % "','".join(str(g) for g in groups)
        sql = """select feat, count(*) from %s %s group by feat"""%(self.featureTable, where)
        if SS:
            mm.executeGetSSCursor(self.corpdb, sql, charset=self.encoding, use_unicode=self.use_unicode, host=self.mysql_host)
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode) 

    def getFeatureCountsSS(self, groupFreqThresh = 0, where = ''):
        """
        Gets feature occurence by group
        
        Args: 
            groupFreqThresh (int): Minimum number of words a group must contain to be considered valid
            where (string): Conditional sql string to limit the search to elements meeting a specified criteria
        Returns:  
            returns a list of (feature, count) tuples, 
            where count is the feature occurence in each group
        """
        return self.getFeatureCounts(groupFreqThresh, where, True)

    def getFeatureValueSums(self, where = ''):
        """returns a list of (feature, count) tuples, where count is the number of groups with the feature"""
        sql = """select feat, sum(value) from %s group by feat"""%(self.featureTable)
        if (where): sql += ' WHERE ' + where
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode) 

    def getDistinctFeatures(self, where=''):
        """returns a distinct list of (feature) tuples given the name of the feature value field (either value, group_norm, or feat_norm)"""
        sql = "select distinct feat from %s"%(self.featureTable)
        if (where): sql += ' WHERE ' + where
        return map(lambda l: l[0], mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode))

    def getFeatureZeros(self, where=''):
        """returns a distinct list of (feature) tuples given the name of the feature value field (either value, group_norm, or feat_norm)"""
        sql = "select feat, zero_feat_norm from %s"%('mean_'+self.featureTable)
        if (where): sql += ' WHERE ' + where
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

    def getValues(self, where = ''):
        """returns a list of (group_id, feature, value) triples"""
        sql = """select group_id, feat, value from %s"""%(self.featureTable)
        if (where): sql += ' WHERE ' + where
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode) 

    def getSumValue(self, where = ''):
        """returns the sume of all values"""
        sql = """select sum(value) from %s"""%(self.featureTable)
        if (where): sql += ' WHERE ' + where
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)[0][0]

    def getSumValuesByGroup(self, where = ''):
        """ """
        sql = """SELECT group_id, sum(value) FROM %s """ % self.featureTable
        if (where): sql += ' WHERE ' + where  
        sql += """ GROUP BY group_id """
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

    def getSumValuesByFeat(self, where = ''):
        """ """
        sql = """SELECT feat, sum(value) FROM %s """ % self.featureTable
        if (where): sql += ' WHERE ' + where  
        sql += """ GROUP BY feat """
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)

    def getGroupNorms(self, where = ''):
        """returns a list of (group_id, feature, group_norm) triples"""
        sql = """SELECT group_id, feat, group_norm from %s"""%(self.featureTable)
        if (where): sql += ' WHERE ' + where
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode) 

    def getValuesAndGroupNorms(self, where = ''):
        """returns a list of (group_id, feature, value, group_norm) triples"""
        sql = """SELECT group_id, feat, value, group_norm from %s"""%(self.featureTable)
        if (where): sql += ' WHERE ' + where
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode) 

    def getGroupNormsForFeat(self, feat, where = '', warnMsg = False):
        """returns a list of (group_id, feature, group_norm) triples"""
        sql = """SELECT group_id, group_norm FROM %s WHERE feat = '%s'"""%(self.featureTable, MySQLdb.escape_string(feat))
        if (where): sql += ' AND ' + where
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, warnMsg, charset=self.encoding, use_unicode=self.use_unicode) 

    def getGroupNormsForFeats(self, feats, where = '', warnMsg = False):
        """returns a list of (group_id, feature, group_norm) triples"""
        if self.use_unicode:
            fCond = " feat in ('%s')" % "','".join(MySQLdb.escape_string(unicode(f)) for f in feats)
        else:
            fCond = " feat in ('%s')" % "','".join(MySQLdb.escape_string(f) for f in feats)
        sql = """SELECT group_id, group_norm FROM %s WHERE %s"""%(self.featureTable, fCond)
        if (where): sql += ' AND ' + where
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, warnMsg, charset=self.encoding, use_unicode=self.use_unicode) 

    def getValuesAndGroupNormsForFeats(self, feats, where = '', warnMsg = False):
        """returns a list of (group_id, feature, group_norm) triples"""
        if self.use_unicode:
            fCond = " feat in ('%s')" % "','".join(MySQLdb.escape_string(unicode(f)) for f in feats)
        else:
            fCond = " feat in ('%s')" % "','".join(MySQLdb.escape_string(f) for f in feats)
        sql = """SELECT group_id, value, group_norm FROM %s WHERE %s"""%(self.featureTable, fCond)
        if (where): sql += ' AND ' + where
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, warnMsg, charset=self.encoding, use_unicode=self.use_unicode) 

    def getValuesAndGroupNormsForFeat(self, feat, where = '', warnMsg = False):
        """returns a list of (group_id, feature, group_norm) triples"""
        if self.use_unicode:
            sql = """SELECT group_id, value, group_norm FROM %s WHERE feat = '%s'"""%(self.featureTable, MySQLdb.escape_string(unicode(feat, 'utf8')))
        else:
            sql = """SELECT group_id, value, group_norm FROM %s WHERE feat = '%s'"""%(self.featureTable, MySQLdb.escape_string(feat))
        if (where): sql += ' AND ' + where
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, warnMsg, charset=self.encoding, use_unicode=self.use_unicode) 


    def getGroupAndFeatureValues(self, featName=None, where=''):
        """returns a list of (group_id, feature_value) tuples"""
        if not featName: featName = self.featNames[0]
        sql = "select group_id, group_norm from %s WHERE feat = '%s'"%(self.featureTable, featName)
        if (where): sql += ' AND ' + where
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, False, charset=self.encoding, use_unicode=self.use_unicode)

    def getGroupsAndFeats(self, where=''):
        fwc.warn("Loading Features and Getting Groups.")
        groups = set()
        features = dict()
        featNames = set(self.featNames)

        for featName in featNames:
            features[featName] = dict(self.getGroupAndFeatureValues(featName, where))
            groups.update(features[featName].keys())

        return (groups, features)

    def getGroupNormsWithZeros(self, groups = [], where = ''):
        """returns a dict of (group_id => feature => group_norm)"""
        #This functino gets killed on large feature sets
        gnlist = []
        if groups: 
            gCond = " group_id in ('%s')" % "','".join(str(g) for g in groups)
            if where: gnlist = self.getGroupNorms(where+" AND "+gCond)
            else: gnlist = self.getGroupNorms(gCond)
        else: 
            gnlist = self.getGroupNorms()
        gns = dict()
        for tup in gnlist:
            (gid, feat, gn) = tup
            if not gid in gns: gns[gid] = dict()
            gns[gid][feat] = gn
        if not groups: groups = self.getDistinctGroups(where)
        allFeats = self.getDistinctFeatures(where)
        #fill in zeros (this can get quite big!)
        fwc.warn("Adding zeros to group norms (%d groups * %d feats)." %(len(groups), len(allFeats)))
        for gid in groups:
            if not gid in gns: gns[gid] = dict()
            for feat in allFeats:
                if not feat in gns[gid]: gns[gid][feat] = 0
        return gns, allFeats

    def getGroupNormsWithZerosFeatsFirst(self, groups = [], where = '', blacklist = None):
        """returns a dict of (feature => group_id => group_norm)"""
        #This functino gets killed on large feature sets
        gnlist = []
        if groups: 
            gCond = " group_id in ('%s')" % "','".join(str(g) for g in groups)
            if where: gnlist = self.getGroupNorms(where+" AND "+gCond)
            else: gnlist = self.getGroupNorms(gCond)
        else: 
            gnlist = self.getGroupNorms()
        gns = dict()
        print "USING BLACKLIST (from getgroupnorms): %s" %str(blacklist)
        for tup in gnlist:
            (gid, feat, gn) = tup
            if blacklist:
                if not any(r.match(feat) for r in blacklist):
                    if not feat in gns: gns[feat] = dict()
                    gns[feat][gid] = gn
            else:
                if not feat in gns: gns[feat] = dict()
                gns[feat][gid] = gn
        if not groups: groups = self.getDistinctGroups(where)
        allFeats = self.getDistinctFeatures(where)
        if blacklist:
            allFeats = list(set(allFeats) - set(blacklist))
        #fill in zeros (this can get quite big!)
        fwc.warn("Adding zeros to group norms (%d groups * %d feats)." %(len(groups), len(allFeats)))
        for feat in allFeats:
            if not feat in gns: gns[feat] = dict()
            thisGn = gns[feat]
            for gid in groups:
                if not gid in thisGn: thisGn[gid] = 0
        return gns, allFeats

    def getGroupNormsSparseFeatsFirst(self, groups = [], where = ''):
        """returns a dict of (feature => group_id => group_norm)"""
        #This functino gets killed on large feature sets
        gnlist = []
        if groups: 
            gCond = " group_id in ('%s')" % "','".join(str(g) for g in groups)
            if where: gnlist = self.getGroupNorms(where+" AND "+gCond)
            else: gnlist = self.getGroupNorms(gCond)
        else: 
            gnlist = self.getGroupNorms()
        gns = dict()
        groups = set()
        for tup in gnlist:
            (gid, feat, gn) = tup
            if not feat in gns: gns[feat] = dict()
            gns[feat][gid] = gn
            groups.add(gid)

        gCond = " group_id in ('%s')" % "','".join(str(g) for g in groups)
        if where: gCond = where+" AND "+gCond
        allFeats = self.getDistinctFeatures(gCond)

        return gns, allFeats

    def yieldGroupNormsWithZerosByFeat(self, groups = [], where = '', values = False, feats = []):
        """yields (feat, groupnorms, number of features"""
        """ or if values = True, (feat, values, groupnorms, number of features)"""
        allFeats = feats
        if not feats: 
            allFeats = self.getDistinctFeatures(where)
        else:
            fwc.warn("feats restricted to %s" % feats)
        
        numFeats = len(allFeats)
        gCond = None
        if groups: 
            gCond = " group_id in ('%s')" % "','".join(str(g) for g in groups)
        else: 
            groups = self.getDistinctGroups(where)
        numGroups = len(groups)

        getGroupNorms = self.getGroupNorms
        getGroupNormsForFeat = self.getGroupNormsForFeat
        getGroupNormsForFeats = self.getGroupNormsForFeats
        if values:
            getGroupNorms = self.getValuesAndGroupNorms
            getGroupNormsForFeat = self.getValuesAndGroupNormsForFeat
            getGroupNormsForFeats = self.getValuesAndGroupNormsForFeats
            
        #figure out if too big for memory:
        fwc.warn("Yielding norms with zeros (%d groups * %d feats)." %(len(groups), numFeats))
        gns = dict()
        vals = dict() #only gets field if values is true
        if (numFeats * numGroups) < 12500000*fwc.GIGS_OF_MEMORY:
            #statically acquire all gns
            gnlist = []
            if gCond: 
                if where: gnlist = getGroupNorms(where+" AND "+gCond)
                else: gnlist = getGroupNorms(gCond)
            else: #don't need to specify groups
                gnlist = getGroupNorms()
            if feats:
                if where:
                    where = " AND ".join([where, "feat IN ('"+"','".join(feats)+"')"])
                else:
                    where = " feat IN ('"+"','".join(feats)+"')"
                
            for tup in gnlist:
                (gid, feat) = tup[0:2]
                if not feat in gns: 
                    gns[feat] = dict()
                    if values:
                        vals[feat] = dict()
                gns[feat][gid] = float(tup[-1])
                if values:
                    vals[feat][gid] = float(tup[2])
        else:
            fwc.warn("Too big to keep gns in memory, querying for each feature (slower, but less memory intensive)")

        def getFeatValuesAndGNs(feat):
            if gns:
                try:
                    if values: 
                        return (vals[feat].copy(), gns[feat].copy())
                    return (None, gns[feat].copy())
                except KeyError:
                    fwc.warn("Couldn't find gns for feat: %s (group_freq_thresh may be too high)" % feat)
                    return (None, dict())
            else:#must query for feat
                gnDict = None 
                valDict = None
                gnlist = []
                if gCond: 
                    if where: gnlist = getGroupNormsForFeat(feat, where+" AND "+gCond)
                    else: gnlist = getGroupNormsForFeat(feat, gCond)
                else:
                    gnlist = self.getGroupNormsForFeat(feat)
                if values:
                    gnDict = dict([(g, float(gn)) for g, _, gn in gnlist])
                    valDict = dict([(g, float(v)) for g, v, _ in gnlist])
                else:
                    gnDict = dict([(g, float(gn)) for g, gn in gnlist])
                return (valDict, gnDict)


        #fill in zeros (this can get quite big!)
        for feat in allFeats:
            (valDict, gnDict) = getFeatValuesAndGNs(feat)
            for gid in groups:
                if not gid in gnDict: #add zeros!
                    gnDict[gid] = 0
                    if values and valDict: valDict[gid] = 0
            if values:
                yield (feat, valDict, gnDict, numFeats)
            else:
                yield (feat, gnDict, numFeats)

    def yieldGroupNormsWithZerosByGroup(self, groups = [], where = '', allFeats = None):
        """returns a dict of (group_id, feature_values)"""
        gnlist = []
        if groups: 
            gCond = " group_id in ('%s')" % "','".join(str(g) for g in groups)
            if where: gnlist = self.getGroupNorms(where+" AND "+gCond)
            else: gnlist = self.getGroupNorms(gCond)
        else: 
            gnlist = self.getGroupNorms()
        gns = dict()
        for tup in gnlist:
            (gid, feat, gn) = tup
            if not gid in gns: gns[gid] = dict()
            gns[gid][feat] = gn
        if not groups: groups = self.getDistinctGroups(where)
        if not allFeats:
            allFeats = self.getDistinctFeatures(where)
        #fill in zeros (this can get quite big!)
        fwc.warn("Yielding norms with zeros for %d groups * %d feats." %(len(groups), len(allFeats)))
        for gid in groups:
            thisGns = dict()
            if gid in gns: thisGns.update(gns[gid])
            for feat in allFeats:
                if not feat in thisGns: thisGns[feat] = 0
            yield (gid, thisGns)

    def yieldValuesWithZerosByGroup(self, groups = [], where = '', allFeats = None):
        """returns a dict of (group_id, feature_values)"""
        valuelist = []
        if groups: 
            gCond = " group_id in ('%s')" % "','".join(str(g) for g in groups)
            if where: valuelist = self.getValues(where+" AND "+gCond)
            else: valuelist = self.getValues(gCond)
        else: 
            valuelist = self.getValues()
        values = dict()
        for tup in valuelist:
            (gid, feat, value) = tup
            if not gid in values: values[gid] = dict()
            values[gid][feat] = value
        if not groups: groups = self.getDistinctGroups(where)
        if not allFeats:
            allFeats = self.getDistinctFeatures(where)
        #fill in zeros (this can get quite big!)
        fwc.warn("Yielding values with zeros for %d groups * %d feats." %(len(groups), len(allFeats)))
        for gid in groups:
            thisValues = dict()
            if gid in values: thisValues.update(values[gid])
            for feat in allFeats:
                if not feat in thisValues: thisValues[feat] = 0
            yield (gid, thisValues)

    def yieldValuesSparseByGroup(self, groups = [], where = '', allFeats = None):
        """returns a dict of (group_id, feature_values)"""
        valuelist = []
        if groups: 
            gCond = " group_id in ('%s')" % "','".join(str(g) for g in groups)
            if where: valuelist = self.getValues(where+" AND "+gCond)
            else: valuelist = self.getValues(gCond)
        else: 
            valuelist = self.getValues()
        values = dict()
        for tup in valuelist:
            (gid, feat, value) = tup
            if not gid in values: values[gid] = dict()
            values[gid][feat] = value
        if not groups: groups = self.getDistinctGroups(where)
        if not allFeats:
            allFeats = self.getDistinctFeatures(where)
        #fill in zeros (this can get quite big!)
        fwc.warn("Yielding values with zeros for %d groups * %d feats." %(len(groups), len(allFeats)))
        for gid in groups:
            thisValues = dict()
            if gid in values: thisValues = values[gid]
            yield (gid, thisValues)

    def printJoinedFeatureLines(self, filename, delimeter = ' '):
        """prints feature table like a message table in format mallet can use"""

        f = open(filename, 'w')
        for (gid, featValues) in self.yieldValuesSparseByGroup():
            message = delimeter.join([delimeter.join([feat.replace(' ', '_')]*value) for feat, value in featValues.iteritems()])
            f.write("""%s en %s\n""" %(gid, message.encode('utf-8')))            
       
        f.close()
        fwc.warn("Wrote joined features file to: %s"%filename)
    
    def getFeatNorms(self, where = ''):
        """returns a list of (group_id, feature, feat_norm) triples"""
        sql = """select group_id, feat, feat_norm from %s"""%(self.featureTable)
        if (where): sql += ' WHERE ' + where
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode) 

    def getFeatNormsSS(self, where = ''):
        """returns a server-side cursor pointing to (group_id, feature, feat_norm) triples"""
        sql = """select group_id, feat, feat_norm from %s"""%(self.featureTable)
        if (where): sql += ' WHERE ' + where
        return mm.executeGetSSCursor(self.corpdb, sql, charset=self.encoding, use_unicode=self.use_unicode, host=self.mysql_host) 

    def getFeatNormsWithZeros(self, groups = [], where = ''):
        """returns a dict of (group_id => feature => feat_norm) """
        fnlist = []
        if groups: 
            gCond = " group_id in ('%s')" % "','".join(str(g) for g in groups)
            if where: gnlist = self.getFeatNorms(where+" AND "+gCond)
            else: fnlist = self.getFeatNorms(gCond)
        else: 
            fnlist = self.getFeatNorms()
        fns = dict()
        for tup in fnlist:
            (gid, feat, fn) = tup
            if not gid in fns: fns[gid] = dict()
            fns[gid][feat] = float(fn)
        if not groups: groups = self.getDistinctGroups(where)

        #fill in zeros (this can get quite big!)
        fwc.warn("Adding zeros to feat norms (%d groups * %d feats)." %(len(groups), len(meanData.keys())))
        meanData = self.getFeatMeanData() # feat : (mean, std, zero_mean)
        for gid in groups:
            if not gid in fns: fns[gid] = dict()
            for feat in meanData.iterkeys():
                if not feat in fns[gid]: fns[gid][feat] = meanData[feat][2] 
        return fns, meanData.keys()

    def getFeatMeanData(self, where = ''):
        """returns a dict of (feature => (mean, std, zero_feat_norm)) """
        meanTable = 'mean_'+self.featureTable
        sql = """select feat, mean, std, zero_feat_norm from %s"""%(meanTable)
        if (where): sql += ' WHERE ' + where
        mList = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode) 
        meanData = dict()
        for tup in mList: #feat : (mean, std, zero_feat_norm)
            meanData[tup[0]] = tup[1:]
        return meanData

    def getContingencyArrayFeatNorm(self, where = ''):
        """ returns a list of lists: each row is a group_id and each col is a feature"""
        """ the first row has a blank first entry and then a list of unique features"""
        """ the first column has a blank first entry and then a list of unique group_ids"""
        fwc.warn("running getContingencyArrayFeatNorm")

        fwc.warn("Getting distinct feature / groupId lists and (feat, featNormZero) list")
        distinctFeatureList = self.getDistinctFeatures( where )
        featureZeroList = self.getFeatureZeros( where )
        distinctGroupList = self.getDistinctGroups( where )

        fwc.warn("Converting feature / groupId lists to dictionaries (item: index) for quick insertion")
        distinctFeatureDict = {}
        counter = 0
        for feature in distinctFeatureList:
            distinctFeatureDict[feature] = counter
            counter += 1

        distinctGroupDict = {}
        counter = 0
        for group in distinctGroupList:
            distinctGroupDict[group] = counter
            counter += 1
        
        fwc.warn("Making a 2d array (matrix) with ncol = nDistinctFeatures and nrow = nDistinctGroupIds")
        fwc.warn("For each distinct feature, intializing that column with feat norm zeros' value")
        contingencyMatrix = zeros( ( len(distinctGroupList), len(distinctFeatureList) ) )
        for tup in featureZeroList:
            (feat, featNormZero) = tup
            columnIndexToZero = distinctFeatureDict[ feat ] 
            contingencyMatrix[ :, columnIndexToZero ] = featNormZero

        fwc.warn("calling getFeatNormsSS, iterating through (with SS cursor)")
        fwc.warn("for each iteration, using the index dictionaries to insert the entry into the matrix")
        ssCursor = self.getFeatNormsSS( where )
        for tup in ssCursor:
            (gid, feat, featNorm) = tup
            columnIndexForInsertion = distinctFeatureDict[ feat ]
            rowIndexForInsertion = distinctGroupDict[ gid ]
            contingencyMatrix[ rowIndexForInsertion, columnIndexForInsertion ] = featNorm

        fwc.warn("returning [contingency matrix, rownames (distinct groups), and colnames (distinct features)]")
        return [ contingencyMatrix, distinctGroupList, distinctFeatureList ]


    def getFeatAll(self, where = ''):
        """returns a list of (group_id, feature, value, group_norm) tuples"""
        sql = """select group_id, feat, value, group_norm from %s"""%(self.featureTable)
        if (where): sql += ' WHERE ' + where
        return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode) 

    def getFeatAllSS(self, where = '', featNorm=True):
        """returns a list of (group_id, feature, value, group_norm) tuples"""
        sql = """select group_id, feat, value, group_norm from %s"""%(self.featureTable) if featNorm else """select group_id, feat, value, group_norm from %s"""%(self.featureTable)
        if (where): sql += ' WHERE ' + where
        return mm.executeGetSSCursor(self.corpdb, sql, charset=self.encoding, use_unicode=self.use_unicode, host=self.mysql_host) 

    def countGroups(self, groupThresh = 0, where=''):
        """returns the number of distinct groups (note that this runs on the corptable to be accurate)"""
        if groupThresh:
            groupCnts = self.getGroupWordCounts(where)
            count = 0
            for wordCount in groupCnts.itervalues():
                if (wordCount >= groupThresh):
                    count += 1
            return count
        else:
            sql = """select count(DISTINCT %s) from %s""" %(self.correl_field, self.corptable)
            if (where): sql += ' WHERE ' + where
            return mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)[0][0]
            
    def getDistinctGroupsFromFeatTable(self, where=""):
        """Returns the distinct group ids that are in the feature table"""
        sql = "select distinct group_id from %s" % self.featureTable
        if (where): sql += ' WHERE ' + where
        return map(lambda l:l[0], mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode))


    def getDistinctGroups(self, where=''):
        """returns the distinct distinct groups (note that this runs on the corptable to be accurate)"""
        sql = """select DISTINCT %s from %s""" %(self.correl_field, self.corptable)
        if (where): sql += ' WHERE ' + where
        return map(lambda l: l[0], mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode))
    
    def ttestWithOtherFG(self, other, maskTable= None, groupFreqThresh = 0):
        """Performs PAIRED ttest on differences between group norms for 2 tables, within features"""
        """to-do: switch for paired ttest or not"""

        #read mask table and figure out groups for each mask:
        masks = {'no mask': set()}
        if maskTable:
            maskList = mm.getTableColumnNameList(self.corpdb, self.dbCursor, maskTable, charset=self.encoding, use_unicode=self.use_unicode)
            print maskList
            assert self.correl_field in maskList, "group field, %s, not in mask table" % self.correl_field
            maskToIndex = dict([(maskList[i], i) for i in xrange(len(maskList))])
            groupIndex = maskToIndex[self.correl_field]

            #get data:
            sql = """SELECT %s FROM %s""" % (', '.join(maskList), maskTable)
            maskData = mm.executeGetList(self.corpdb, self.dbCursor, sql, charset=self.encoding, use_unicode=self.use_unicode)
            for maskId in maskList:
                if not maskId == self.correl_field:
                    masks[maskId] = set()
            for row in maskData: 
                groupId = row[groupIndex]
                for i in xrange(len(row)):
                    if i != groupIndex and row[i] == 1:
                        masks[maskList[i]].add(groupId)

        #apply masks
        results = dict() #mask => results
        for mid, mask in masks.iteritems():

            threshGroups1 = set()
            threshGroups2 = set()

            # get groups passing GFT for BOTH 
            if groupFreqThresh:
                print 'groupFreqThresh set to '+str(groupFreqThresh)
                groupCnts1 = self.getGroupWordCounts(lexicon_count_table=self.getWordTable(self.featureTable.split('$')[2]))
                #print groupCnts1
                for group, wordCount in groupCnts1.iteritems():
                    if (wordCount >= groupFreqThresh):
                        threshGroups1.add(group)
                groupCnts2 = other.getGroupWordCounts(lexicon_count_table=other.getWordTable(other.featureTable.split('$')[2]))
                #print groupCnts2
                for group, wordCount in groupCnts2.iteritems():
                    if (wordCount >= groupFreqThresh):
                        threshGroups2.add(group)

            print str(len(threshGroups1))+' groups pass groupFreqThresh for feat table 1'
            print str(len(threshGroups2))+' groups pass groupFreqThresh for feat table 2'
            threshGroups = threshGroups1 & threshGroups2
            if mask: 
                threshGroups = threshGroups & mask
            threshGroups = list(threshGroups)
            print str(len(threshGroups))+' groups pass groupFreqThresh for BOTH'
            assert len(threshGroups) > 0, "No groups passing frequency threshold"

            #find features:
            feats1 = self.getDistinctFeatures()
            feats2 = other.getDistinctFeatures()
            featsInCommon = list(set(feats1) & set(feats2))

            ttestResults = dict()

            featYielder1 = self.yieldGroupNormsWithZerosByFeat(groups = threshGroups, feats = featsInCommon)
            featYielder2 = other.yieldGroupNormsWithZerosByFeat(groups = threshGroups, feats = featsInCommon)

            for (feat1, dataDict1, Nfeats1) in featYielder1:
                (feat2, dataDict2, Nfeats2) = featYielder2.next()

                assert feat1==feat2, 'feats do not match'
                assert sorted(dataDict1)==sorted(dataDict2), 'groups do not match'

                gns1 = [gn for (group, gn) in sorted(dataDict1.items())]
                gns2 = [gn for (group, gn) in sorted(dataDict2.items())]

                #t,p = ttest_rel(gns1,gns2)
                t,p, d = self.pairedTTest(gns1,gns2)
                ttestResults[feat1] = {'t': t, 'p': p, 'd': d, 'N': len(gns1)}
        
            results[mid] = ttestResults # dict for each feat

        return results

    # pandas dataframe methods
    def getGroupNormsAsDF(self, where=''):
        """returns a dataframe of (group_id, feature, group_norm)"""
        """default index is on group_id and feat"""
        index=['group_id','feat']
        db_eng = mif.get_db_engine(self.corpdb)
        sql = """SELECT group_id, feat, group_norm from %s""" % (self.featureTable)
        if (where): sql += ' WHERE ' + where
        return pd.read_sql(sql=sql, con=db_eng, index_col=index)

    def getValuesAsDF(self, where=''):
        """returns a dataframe of (group_id, feature, value)"""
        """default index is on group_id and feat"""
        index=['group_id','feat']
        db_eng = mif.get_db_engine(self.corpdb)
        sql = """SELECT group_id, feat, value from %s""" % (self.featureTable)
        if (where): sql += ' WHERE ' + where
        return pd.read_sql(sql=sql, con=db_eng, index_col=index)

    def getGroupNormsWithZerosAsDF(self, groups=[], where='', pivot=False, sparse=False):
        """returns a dict of (group_id => feature => group_norm)"""
        """default index is on group_id and feat"""
        index=['group_id','feat']
        db_eng = mif.get_db_engine(self.corpdb)
        sql = """SELECT group_id, feat, group_norm from %s""" % (self.featureTable)
        if groups:
            gCond = " group_id in ('%s')" % "','".join(str(g) for g in groups)
            if (where): sql += ' WHERE ' + where + " AND " + gCond
            else: sql += ' WHERE ' + gCond
        elif (where):
            sql += ' WHERE ' + where
        if pivot:
            if sparse:
                return pd.read_sql(sql=sql, con=db_eng, index_col=index).unstack().to_sparse().fillna(value=0)
            else:
                return pd.read_sql(sql=sql, con=db_eng, index_col=index).unstack().fillna(value=0)
        else:
            # this method won't work if default index is changed
            df =  pd.read_sql(sql=sql, con=db_eng, index_col=index)
            idx = pd.MultiIndex.from_product([df.index.levels[0], df.index.levels[1]], names=df.index.names)
            if sparse:
                return df.reindex(idx).to_sparse().fillna(value=0)
            else:
                return df.reindex(idx).fillna(value=0)

    def getValuesAndGroupNormsAsDF(self, where=''):
        """returns a dataframe of (group_id, feature, value, group_norm)"""
        """default index is on group_id and feat"""
        index=['group_id','feat']
        db_eng = mif.get_db_engine(self.corpdb)
        sql = """SELECT group_id, feat, value, group_norm from %s""" % (self.featureTable)
        if (where): sql += ' WHERE ' + where
        return pd.read_sql(sql=sql, con=db_eng, index_col=index)

    @staticmethod
    def pairedTTest(y1, y2):
        y1, y2 = array(y1), array(y2)
        n = len(y1)
        y_diff = y1 - y2
        y_diff_mean, yfcra_sd = mean(y_diff), std(y_diff)
        t = y_diff_mean / (yfcra_sd / sqrt(n))
        p = spt.sf(np.abs(t), n-1)
        y1_mean, y1_std = mean(y1), std(y1)
        y1_y1z = (y1 - y1_mean) / y1_std
        y2_y1z = (y2 - y1_mean) / y1_std
        #assert mean(y1_y1z) == 0.000, "y1 mean not zero, %.5f" % mean(y1_y1z) #will be close enough to zero
        d = mean(y2_y1z)
        return (t, p, d)
