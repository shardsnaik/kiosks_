from main.components.llm_engine_.configs.config import cashier_configs
from main.components.llm_engine_.cashier_root import llm_engine

class llm_maneger_pipeline:

    def __init__(self):
        '''
        intiating configuration of llm config
        '''
        configs = cashier_configs()
        self.all_cahier_configs =configs.cashier_all_configs()
        self.obj = llm_engine(self.all_cahier_configs, self.all_cahier_configs)
        self.model, self.tokenizer,self.embedder, self.text_embeddings, self.text_chunks = self.obj.load_model_and_doc()
    
    def main(self)-> str:
        '''
        the main function of llm manager pipeline returns assistant responce as output which is nessary for extacting the intent

        Args:
            text genrated form speech recognition pipeline

        Returns:
            str: the response from llm engine 
        '''
        
        question, response = self.obj.llm_core(self.model, self.tokenizer,self.embedder, self.text_embeddings, self.text_chunks)
        # if response is None:
        #     response = "I'm sorry, I didn't understand that. Could you please rephrase your question?"
        #     return question, response
        return question, response


# if __name__ == '__main__':
#     obj = llm_maneger_pipeline()
#     obj.main()

