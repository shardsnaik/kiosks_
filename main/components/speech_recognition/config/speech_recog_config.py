from main.functions.common_function import read_yaml_file, create_directories
from pathlib import Path
class speech_recog_config:
    def __init__(self):
        self.config = read_yaml_file(Path('config/config.yaml'))
        self.params = read_yaml_file(Path('config/params.yaml'))

    def speech_recog_module(self):
        config = self.config.local_models
        create_directories([config.speech_recog_model])

        sppech_recog_config = {
            'model_path': Path(config['speech_recog_model']),
            'model': config['speech_to_text_model'],
            'duration': self.params['duration'],
            'sample_rate': self.params['sampling_rate'],
            'channel': self.params['channel'],
        }
        
        return sppech_recog_config
        
