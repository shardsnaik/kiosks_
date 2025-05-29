from main.functions.common_function import *

class chat_manager_config:
    def __init__(self):
        self.configs = read_yaml_file(Path('config/config.yaml'))
        self.params = read_yaml_file(Path('config/params.yaml'))

    def chat_manager_configs(self):
        """
        Returns the chat manager configurations 
        
        Returns:
             dict: A dictionary containing chat manager configurations
             
        """
        config = self.configs.chat_history
        create_directories([config.chat_history_path])

        all_chat_manager_configs = {
            'chat_history_path': Path(config['chat_history_path']),
            'chat_history_file_name': config['chat_history_file_name']
        }
        return all_chat_manager_configs