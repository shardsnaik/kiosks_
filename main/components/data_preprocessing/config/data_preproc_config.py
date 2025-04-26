from main.functions.common_function import read_yaml_file, create_directories
from pathlib import Path

class data_preproc_configs:

    def __init__(self):
        self.config = read_yaml_file(Path('config/config.yaml'))
        self.params = read_yaml_file(Path('config/params.yaml'))

    def data_preprocessing_configs(self):
        config = self.config.artifacts
        create_directories([config.datasets])

        data_preprocess_comfigurartions ={
            'dataset_path': Path(config['datasets']),
            'raw_dataset_path': Path(config['raw_datasets'])
        }

        return data_preprocess_comfigurartions