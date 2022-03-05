from typing import Optional, List
import json

######################

def prune_transformer_layers(model):
    return

class sentence_tokenizer:
    """
        NLTK sentence tokenizer
    """
    def __init__(self):
        try:
            import nltk.data
            import sys
        except ImportError:
             print("warning: unable to import nltk.tree or nltk.corpus or nltk.data")

        self.sentDetector = nltk.data.load('tokenizers/punkt/english.pickle')

    def __call__(self, messageRows):
        
        messages = list(map(lambda x: x[1], messageRows))
        parses = []
        for m_id, message in messageRows:
            parses.append([m_id, json.dumps(self.sentDetector.tokenize(tc.removeNonUTF8(tc.treatNewlines(message.strip()))))])
        return parses
        

class transformer_embeddings:
    """
    Class to retrieve transformer embeddings. Supply table name, model name, layer numbers, aggregation methods, [device numbers], [output table name]
    """
    #modelName, tokenizerName, modelClass=None, batchSize=dlac.GPU_BATCH_SIZE, aggregations = ['mean'], layersToKeep = [8,9,10,11], maxTokensPerSeg=255, noContext=True, layerAggregations = ['concatenate'], wordAggregations = ['mean'], keepMsgFeats = False, customTableName = None, valueFunc = lambda d: d
    
    def __init__(self, tableName:str, modelName:str, tokenizerName:str=None, layersToKeep:List=[-1, -2, -3, -4], aggregations:List=['mean'], layerAggregations:List=['mean'], wordAggregations:List=['mean'], maxTokensPerSeg=None, batchSize:int=dlac.GPU_BATCH_SIZE, noContext=True, customTableName:str=None, savePath:str=None):
        
        self.groups_processed = 0
        self.groups = [] #run get_groups()
        self.batchSize = batchSize   
        
    def load_text(self, ):
        """
        Function to load text from the database
        """
        groups_to_process = self.groups[self.groups_processed:]

    def addSentTokenized(self, messageRows):

        return

    def generate(self, ):
        return

    def aggregate(self, ):

    def msg_level_agg(self,):
        return

    def layer_level_agg(self,):
        return 

    def group_level_agg(self,):
        return

    def save_embeddings(self):
