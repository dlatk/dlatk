######################
######################
#author-gh: @adithya8
#HaRT embeddings for DLATK
######################

import sys
from typing import Optional, List
import json
from json import loads
#from simplejson import loads
import numpy as np
try:
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoConfig, AutoTokenizer, AutoModel
    #TODO: assert transformer minimum version
except ImportError:
    print ("transformers library not present. Install transformers (github.com/huggingface/transformers) to run transformer embeddings command")
    sys.exit()

from . import textCleaner as tc

HART_PATH = "" #path to HaRT Codebase/
try:
    sys.path.append(HART_PATH)
    #Import startment for Hart goes here 
except:
    print ("Error while importing HaRT codebase. Please check the path to HaRT codebase")
    sys.exit()

######################
######################

class sentenceTokenizer:
    """
    Initialize the sentence tokenizer (or any other pre-transformer text processing) inside the __init__ function.
    """
    def __init__(self):

        """
        
        Please include any custom imports for this class inside the try and except block.  
        """
        try:
            import sys
        except ImportError:
             print("warning: unable to import nltk.tree or nltk.corpus or nltk.data")

    def __call__(self, messageRows):
        #Input: List of (msg Id, msg) pairs
        
        return messageRows
        
######################
######################
    
class textTransformerInterface:

    def __init__(self, transformerTokenizer):
        """
        Initialize the transformer tokenizer model inside the __init__ function along with other tokenization relevant objects.
        """
        
    def context_preparation(self, groupedMessageRows):
        """

        """
        #groupedMessageRows: List[correl field id, List[List[Message Id, message]]]
        
        input_ids = []
        token_type_ids = []
        attention_mask = []
        msgId_seq = []
        cfId_seq = []

        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}, (cfId_seq, msgId_seq)

    def no_context_preparation(self, messageRows, sent_tok_onthefly):

        return 

######################
######################

class transformer_embeddings:
    """
    Class to retrieve transformer embeddings. Supply table name, model name, layer numbers, aggregation methods, [device numbers], [output table name]
    """
    #modelName, tokenizerName, modelClass=None, batchSize=dlac.GPU_BATCH_SIZE, aggregations = ['mean'], layersToKeep = [8,9,10,11], maxTokensPerSeg=255, noContext=True, layerAggregations = ['concatenate'], wordAggregations = ['mean'], keepMsgFeats = False, customTableName = None, valueFunc = lambda d: d
    
    def __init__(self, modelName:str, tokenizerName:str=None, layersToKeep:List=[-1, -2, -3, -4], aggregations:List=['mean'], layerAggregations:List=['mean'], wordAggregations:List=['mean'], batchSize:int=None, savePath:str=None):
        """
        modelName: Name of the transformer model to use.
        tokenizerName: Name of the tokenizer to use. If None, defaults to the same as modelName.
        layersToKeep: List of layers to keep
        aggregations: List of aggregations to perform at the layer level
        layerAggregations: List of aggregations to perform at the group level
        wordAggregations: List of aggregations to perform at the word level
        batchSize: Batch size. If None, defaults to the maximum allowed by the GPU.
        savePath: Path to save the embeddings to. If None, defaults to the current directory.
        """
        self.batchSize = batchSize
        self.config = AutoConfig.from_pretrained(modelName, output_hidden_states=True)
        if tokenizerName is not None:
            self.transformerTokenizer = AutoTokenizer.from_pretrained(tokenizerName)
        else:
            self.transformerTokenizer = AutoTokenizer.from_pretrained(modelName)
        #TODO: Add default cache file dir
        self.transformerModel = AutoModel.from_pretrained(modelName, config=self.config)

        #Add relelvant init variables here
        #For example: Down below, fix for gpt2
        self.pad_token_id = self.transformerTokenizer.pad_token_id if self.transformerTokenizer.pad_token_id else 0
        self.transformerModel.eval()

        self.cuda = True
        #TODO: Turn this into dataparallel to leverage multiple GPUs
        try:
            self.transformerModel.to('cuda')
        except:
            print (" unable to use CUDA (GPU) for BERT")
            self.cuda = False

        layersToKeep = self.parse_layers(layersToKeep)
        self.layersToKeep = np.array(layersToKeep, dtype='int')

        self.aggregations = aggregations
        self.layerAggregations = layerAggregations
        self.wordAggregations = wordAggregations #not being used right now
        
        self.textToTokensInterface = textTransformerInterface(self.transformerTokenizer)

    def parse_layers(self, layers):
        """
            Checks the validity of the input layer number
            Turns negative layer idx into equivalent positive and sorts layer numbers in ascending order
        """
        for lyr in layers:
            if (lyr > self.transformerModel.config.num_hidden_layers):
                print (f"You have supplied a layer number ({lyr}) greater than the total number of layers ({self.transformerModel.config.num_hidden_layers}) in the model. Retry with valid layer number(s).")
                sys.exit()

        layers = [(lyr%self.transformerModel.config.num_hidden_layers)+1 if lyr<0 else lyr for lyr in layers]
        #removing duplicate layer inputs
        layers = sorted(list(set(layers)))

        return layers

    def prepare_messages(self, groupedMessageRows ):

        tokenIdsDict, (cfId_seq, msgId_seq) = self.textToTokensInterface.context_preparation(groupedMessageRows)
            
        #print ("Len cfs/msgs: ", len(cfId_seq), len(msgId_seq))
        #print ("Num unique cfs/message Ids: ", np.unique(cfId_seq), len(set(map(lambda x: x[0], msgId_seq))))
        return tokenIdsDict, (cfId_seq, msgId_seq)

    def generate_transformer_embeddings(self, tokenIdsDict):

        #Number of Batches
        num_batches = int(np.ceil(len(tokenIdsDict["input_ids"])/float(self.batchSize)))
        encSelectLayers = []
        #print ('len(input_ids): ',len(tokens["input_ids"]))
        print ('Num Batches:', num_batches)

        if len(tokenIdsDict["input_ids"]) == 0:
            print ("No messages in this batch!!! Message table might be consisting NULLs")
            return encSelectLayers
        
        #TODO: Check if len(messageSents) = 0, skip this and print warning
        for i in range(num_batches):
            #Append embeddings to encSelectLayers
            pass
            
        return encSelectLayers
    
    def _aggregate(self):
        """
        Aggregates the embeddings at the layer, group, and word level. Helper function for aggregate.
        """
        
        return 

    def aggregate(self):
        """
        Function to aggregate the embeddings
        """

        
        return #(cf_reps, cfIds)

    def save_embeddings(self):
        return

######################
######################
