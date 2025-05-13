from main.functions.common_function import read_yaml_file, create_directories
from pathlib import Path

class cashier_configs:

    def __init__(self):
        self.config = read_yaml_file(Path('config/config.yaml'))
        self.params = read_yaml_file(Path('config/params.yaml'))

    def cashier_all_configs(self):
        config = self.config.cashier
        create_directories([config.base_model_path])

        all_cahier_configs = {
            'base_model_path': Path(config['base_model_path']),
            'base_model_id': config['base_model_id'],
            'dowloaded_base_model_path':  Path(config['base_model_path']) / config['base_model_id'],
            'fine_tuned_model_id' :config['fine_tuned_model'],
            'rag_file_in_json' :config['rag_file_in_json']
        }

        return all_cahier_configs