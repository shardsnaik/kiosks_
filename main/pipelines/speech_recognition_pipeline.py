from main.components.speech_recognition.config.speech_recog_config import speech_recog_config
from main.components.speech_recognition.speech_recog import speech_recog_compo


class speech_recognition_pipeline:
    def __init__(self):
        obj = speech_recog_config()
        self.config_params = obj.speech_recog_module()


    def main(self):
        print(" 🎙️🎙️ Initializing Speech Recognition Pipeline...")
        print("✔️ Loading configuration...")
        component = speech_recog_compo(self.config_params, self.config_params)
        speech_input = component.record_and_transcribe_audio()
        print(type(speech_input))
        return speech_input


# if __name__ == '__main__':
#     obj = speech_recognition_pipeline()
#     obj.main()
