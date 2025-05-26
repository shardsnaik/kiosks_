from main.components.TTS.config.tts_config import tts_configration

import pyttsx3

class TTS:
    def __init__(self, config, params):
        self.config = config
        self.params = params

    def tts_root(self, gen_text:str):
        engine = pyttsx3.init()
        engine.say(gen_text)
        engine.runAndWait()


