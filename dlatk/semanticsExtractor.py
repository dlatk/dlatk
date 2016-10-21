"""
Semantic Extractor

Bridges DLATK's Feature Extractor with reading xml semantic annotations
assumes that corptable is a directory with files rather than a database table. 
"""
import os
import gzip
import sys
import xml.etree.cElementTree as ET

from pprint import pprint

from .featureExtractor import FeatureExtractor
from .outcomeGetter import OutcomeGetter
from . import featureWorker

#############################################################
## Static Variables and Methods
#
COLLOC_LENGTH = 96

def _warn(string):
    print(string, file=sys.stderr)

class SemanticsExtractor(FeatureExtractor):

    def __init__(self, corpdb, corptable, correl_field, mysql_host, message_field, messageid_field, corpdir):
        #corptable = "_".join(corpdir.split('/')[-2:])
        super(FeatureExtractor, self).__init__(corpdb, corptable, correl_field, mysql_host, message_field, messageid_field)
        self.corpdir = corpdir


    def addNERTable(self, tableName = None, min_freq=1, valueFunc = lambda d: d, normalizeByCollocsInGroup = False):
        """extracts named entiry features from semantic xml files"""
        #assumes each file is one group
        #normalizeByCollocsInGroup: if true, will use the number of collocs as the value to normalize by rather than terms 

        ##CREATE TABLES##
        tableNames = {'ner': self.createFeatureTable('ner', "VARCHAR(%d)"%int(COLLOC_LENGTH/3), 'INTEGER', tableName, valueFunc),
                      'ner_colloc': self.createFeatureTable('ner_colloc', "VARCHAR(%d)"%int(COLLOC_LENGTH*1.333), 'INTEGER', tableName, valueFunc),
                      'colloc':  self.createFeatureTable('colloc', "VARCHAR(%d)"%COLLOC_LENGTH, 'INTEGER', tableName, valueFunc)}

        ##FIND FILES##
        corpdir = self.corpdir
        subdirs = os.listdir(corpdir)
        groupsInserted = 0
        if len(subdirs) > 50: [self._disableTableKeys(name) for name in list(tableNames.values())]#for faster, when enough space for repair by sorting
        for subdir in subdirs:
            #print "on %s" % subdir #debug

            files = os.listdir(corpdir+'/'+subdir)
            for filename in files:
                f = gzip.open(corpdir+'/'+subdir+'/'+filename, 'rb')
                xmlStr = f.read()
                f.close()
                (_, _, xmlStr) = xmlStr.partition("\n")
                xmlRoot = None
                try: 
                    xmlRoot = ET.fromstring(xmlStr)
                except ET.ParseError:
                    #skip if xml is malformed
                    continue

                #setup group specific variables
                freqs = {'ner': dict(),
                         'colloc': dict(),
                         'ner_colloc': dict()}
                totalTermsInGroup = 0
                totalCollocsInGroup = 0
                for document in xmlRoot.findall('document'):
                    for parsed in document.findall('parsed'):
                        for section in parsed.findall('section'):
                            for sent in section.findall('sentence'):
                                #at this point we have terms and nonterms, nonterms group terms together
                                for nonterm in sent.findall('nonterm'):
                                    #add to totalTokens
                                    totalTermsInGroup += len(nonterm.findall('term'))
                                    totalCollocsInGroup += 1
                                    #get ne class + colloc
                                    (ne, colloc) = ('','')
                                    for attr in nonterm.findall('attr'):
                                        if attr.attrib['name'] == 'ne':
                                            ne = attr.attrib['value']



                                        elif attr.attrib['name'] == 'nealias':
                                            colloc = attr.attrib['value']
                                    try:
                                        freqs['ner'][ne] += 1
                                    except KeyError:
                                        freqs['ner'][ne] = 1
                                    try:
                                        freqs['colloc'][colloc] += 1
                                    except KeyError:
                                        freqs['colloc'][colloc] = 1
                                    try:
                                        freqs['ner_colloc']["%s:%s"%(ne,colloc)] += 1
                                    except KeyError:
                                        freqs['ner_colloc']["%s:%s"%(ne,colloc)] = 1

                                totalTermsInGroup += len(nonterm.findall('term'))
                                    
                #pprint(freqs['ner_colloc']) #debug
                #print totalTermsInGroup #debug

                ##INSERT INTO DB##
                group_id = filename.split('.')[0] #first part of filename is group id
                #print group_id #debug
                total = float(totalTermsInGroup)
                if normalizeByCollocsInGroup:
                    total = float(totalCollocsInGroup)
                for dataType, featFreqs in freqs.items():
                    wsql = """INSERT INTO """+tableNames[dataType]+""" (group_id, feat, value, group_norm) values ('"""+str(group_id)+"""', %s, %s, %s)"""
                    rows = [(k.encode('utf-8'), v, valueFunc((v / total))) for k, v in featFreqs.items() if v >= min_freq] #adds group_norm and applies freq filter
                    self._executeWriteMany(wsql, rows)

                groupsInserted+=1
                if groupsInserted % 1000 == 0:
                    _warn("\n\t%d groups inserted" % groupsInserted)

                #raise NotImplementedError

        if len(subdirs) > 50: 
            _warn("Adding Keys (if goes to keycache, then consider running myisamchk -n).")
            [self._enableTableKeys(name) for name in list(tableNames.values())]#for faster, when enough space for repair by sorting

        return tableNames['ner'] #feature creation methods are expected to return 1 table name so I just chose NER
