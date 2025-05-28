from main.components.chat_manager.config.chat_manager_config import chat_manager_config

class ChatManager:
    def __init__(self, config, params):
        self.config = config
        self.params = params

    def chat_history_initiator(self):
        