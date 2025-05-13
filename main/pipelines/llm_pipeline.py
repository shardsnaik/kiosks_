from main.components.llm_engine_.configs.config import cashier_configs
from main.components.llm_engine_.cashier_root import llm_engine

class llm_maneger_pipeline:

    def __init__(self):
        '''
        intiating configuration of llm config
        '''
        configs = cashier_configs()
        self.all_cahier_configs =configs.cashier_all_configs()
    
    def llm_engine(self):
        obj = llm_engine()
        model, tokenizer,embedder, text_embeddings, text_chunks = obj.load_model_and_doc()