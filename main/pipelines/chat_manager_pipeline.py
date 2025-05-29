from main.components.chat_manager.config.chat_manager_config import chat_manager_config
from main.components.chat_manager.chat_manager import ChatManager

class ChatManagerPipeline:

    def __init__(self):
        pass

    def chat_history_main(self, user_text, bot_text):
        obj = chat_manager_config()
        config_params = obj.chat_manager_configs()

        # Calling the component function
        chat_manager = ChatManager(config_params, config_params)
        chat_manager.chat_history_initiator(user_text, bot_text)

