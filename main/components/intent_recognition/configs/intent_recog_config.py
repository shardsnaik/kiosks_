from main.functions.common_function import save_json_file, create_directories, read_yaml_file

class intent_recogntion_configs:
    def __init__(self):
        self.config = read_yaml_file('config/config.yaml')
        self.params = read_yaml_file('config/params.yaml')
    
    def intent_recogntion_all_configs(self):
        config = self.config.intent_recog
        params = self.params['intent_keywords']
        create_directories([config['nltk_path']])

        intent_recogntion_configrations ={
            'nltk_path': config['nltk_path'],
            'intent_keywords': params.ordering_intent
        
        }
        return intent_recogntion_configrations