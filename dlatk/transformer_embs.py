######################

def prune_transformer_layers(model):
    return

class transformer_embeddings():
    """
    Class to retrieve transformer embeddings. Supply table name, model name, layer numbers, aggregation methods, [device numbers], [output table name]
    """
    def __init__(self, table_name: str, model_name_or_path: str, batch_size: int = 32, emb_table_name: str = None, save_path: str = None):
        self.groups_processed = 0
        self.groups = [] #run get_groups()
        self.batch_size = batch_size
        
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
