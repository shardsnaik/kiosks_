from main.pipelines.menu_maneger_pipeline import MenuManagerPipeline
from main.pipelines.speech_recognition_pipeline import speech_recognition_pipeline
from main.pipelines.llm_pipeline import llm_maneger_pipeline
from main.pipelines.intent_recognition_pipeline import intent_recognition_pipeline
from main.pipelines.tts_pipeline import tts_pipeline
from main.pipelines.chat_manager_pipeline import ChatManagerPipeline

class  the_AI_Kiosks:

    def __init__(self):
        self.intent_recog_obj = intent_recognition_pipeline()
        self.tts_obj = tts_pipeline()
        self.history_obj =  ChatManagerPipeline()

    def run_main(self):
        # # Initiating menu manager pipeline
        # obj = MenuManagerPipeline()
        # obj.run_pipeline()
        
        # Initiating speech recognition pipeline
        # speech_recog_obj = speech_recognition_pipeline()
        # speech_text = speech_recog_obj.main()

        # Initiating LLM manager pipeline ( which is internally connected with speech recognition pipeline) 
        llm_obj = llm_maneger_pipeline()
        while True:
            ques, res = llm_obj.main()

            # Initiating intent recognition pipeline 
            '''
            res is the text generated from llm  and the user voice input both are used to extract the intent
            '''
            self.intent_recog_obj.intent_recog_main(ques)
            self.tts_obj.main(res)

            # pipeline for saving the chat history
            self.history_obj.chat_history_main(user_text=ques, bot_text=res)

            if ques in ['exit', 'quit']:
                break




if __name__ == '__main__':
    kiosks = the_AI_Kiosks()
    kiosks.run_main()

