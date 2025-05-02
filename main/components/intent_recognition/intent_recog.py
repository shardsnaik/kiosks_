from main.components.intent_recognition.configs.intent_recog_config import intent_recogntion_configs
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


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
        lemmatizer = WordNetLemmatizer()
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
    
        # Lemmatize tokens
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # extracting the intent from params.yaml
        # print(self.params['intent_keywords']) 
        
        text_lower = text.lower()
        intent_matches = {}        
        for intent, phrases in self.params['intent_keywords'].items():
            matches = sum(1 for phrase in phrases if phrase in text_lower)
            intent_matches[intent] = matches

        if max(intent_matches.values(), default=0) > 0:
            return max(intent_matches.items(), key=lambda x: x[1])[0]
        else:
            return "unknown_intent"

text =[
    'just add two biryani',
    'i want to cancel my order',
    'remove one biryani'
]

if __name__ == '__main__':
    print('<<<<<<<< ❕ Interfering INTENT RECOGNITION Component ‼️ >>>>>>>>>>>')
    config_obj = intent_recogntion_configs()
    config_params = config_obj.intent_recogntion_all_configs()
    
    objs = intent_recognition_compo(config_params,config_params)
    # print(objs.extract_intent_with_nltk("just cancel my order i don't want it anaymore"))
    for num,i in enumerate(text):
        print(f'{num+1}.{i} => {objs.extract_intent_with_nltk(i)}')


