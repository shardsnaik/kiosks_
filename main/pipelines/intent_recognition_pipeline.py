from main.components.intent_recognition.configs.intent_recog_config import intent_recogntion_configs
from main.components.intent_recognition.intent_recog import intent_recognition_compo

class intent_recognition_pipeline:
    def __init__(self):
        '''
        Initiating configuration of intent recognition component
        '''
        config_obj = intent_recogntion_configs()
        self.config_params = config_obj.intent_recogntion_all_configs()
        self.obj = intent_recognition_compo(self.config_params, self.config_params)

    def intent_recog_main(self, text: str) -> str:
        '''
        The main function of intent recognition pipeline returns the intent extracted from the text

        Args:
            text (str): The input text for which the intent needs to be recognized

        Returns:
            str: The recognized intent
        '''
        ans =  self.obj.extract_intent_with_nltk(text)
        print(f'intent_recognzed =>{ans}')
