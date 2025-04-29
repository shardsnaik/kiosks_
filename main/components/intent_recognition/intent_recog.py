from main.components.intent_recognition.configs.intent_recog_config import intent_recogntion_configs
import nltk

class intent_recognition_compo:
    def __init__(self, config, params):
        self.config = config
        self.params = params

    def extract_intent_with_nltk(self, text):
        if len(self.config['nltk_path']) ==0:
            # downlaoding nltk data if not already downloaded
            print("Downloading NLTK file beacuse the directory =>", self.config['nltk_path'], "is empty")
            nltk.download(download_dir=self.config['nltk_path'])
            
        nltk.data.path.append(self.config['nltk_path'])
        print(self.params['intent_keywords'])


if __name__ == '__main__':
    print('<<<<<<<< ❕ Interfering INTENT RECOGNITION Component ‼️ >>>>>>>>>>>')
    config_obj = intent_recogntion_configs()
    config_params = config_obj.intent_recogntion_all_configs()
    
    
    objs = intent_recognition_compo(config_params,config_params)
    print(objs.extract_intent_with_nltk("Heloo"))