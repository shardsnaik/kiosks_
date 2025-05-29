from main.functions.common_function import *
import datetime, os

class ChatManager:
    __session_timestamp = None

    def __init__(self, config, params):
        self.config = config
        self.params = params
        # self.history_file = os.path.join(self.config['chat_history_path'], self.config['chat_history_file_name'])
        if ChatManager.__session_timestamp is None:
            time_stamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            ChatManager.__session_timestamp =  f'{time_stamp}.json'
        self.history_file = os.path.join(self.config['chat_history_path'], ChatManager.__session_timestamp)
        os.makedirs(self.config['chat_history_path'], exist_ok=True)


    def chat_history_initiator(self, user_text, bot_text):
        '''
        Initiates the chat history with user and bot text.
        Args:
            user_text (str): The text input from the user.
            bot_text (str): The text response from the llm
        Returns:
            json: A JSON object containing the chat history.
        '''
        log_data = {
            'user_text': user_text.strip(),
            'llm_response': bot_text.strip()
        }

        # Step 1: Load existing history if it exists
        # os.makedirs(self.history_file, exist_ok = True)
        
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    if isinstance(history, dict):
                        history = [history]
            except json.JSONDecodeError:
                history = []
        else:
            history = []

        # Step 2: Append new entry
        history.append(log_data)

        # output_path = 
        save_json_file(self.history_file, history)

        print(f"✅ Saved conversation to {self.history_file}")




