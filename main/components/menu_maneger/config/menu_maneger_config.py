from pathlib import Path
from main.functions.common_function import read_yaml_file, create_directories
class pdf_source:
    '''
    class to manage the source of the pdf files

    takes the source directory and the source URL as input
    and creates the source directory if it does not exist
    '''
    def __init__(self):
        self.config = read_yaml_file(Path('config/config.yaml'))
        

    def create_source_dir(self):
        config = self.config.pdf_maneger
        create_directories([config.pdf_dir])

        pdf_urls = {
            'pdf_dir': Path(config['pdf_dir']),
            'pdf_url' : config['pdf_path']
        }

        return pdf_urls

