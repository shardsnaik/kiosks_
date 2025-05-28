from main.pipelines.menu_maneger_pipeline import MenuManagerPipeline
from main.pipelines.speech_recognition_pipeline import speech_recognition_pipeline
from main.pipelines.llm_pipeline import llm_maneger_pipeline
from main.pipelines.intent_recognition_pipeline import intent_recognition_pipeline
from main.pipelines.tts_pipeline import tts_pipeline

class  the_AI_Kiosks:

    def __init__(self):
        self.intent_recog_obj = intent_recognition_pipeline()
        self.tts_obj = tts_pipeline()

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
            if res == None:
                break




if __name__ == '__main__':
    kiosks = the_AI_Kiosks()
    kiosks.run_main()