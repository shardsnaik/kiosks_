from main.functions.common_function import *
from pathlib import Path

class tts_configration:

    def __init__(self):
        self.config = read_yaml_file(Path('config/config.yaml'))
        self.params = read_yaml_file(Path('config/params.yaml'))

    def tts_configuration(self):
        config = self.config.local_models
        create_directories([])

        tts_config ={
          'model': config['speech_to_text_model'],
            'duration': self.params['duration']  
        }

        return tts_config