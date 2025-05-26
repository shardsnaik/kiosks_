# Importing the TTS configuration class
from main.components.TTS.config.tts_config import tts_configration

# Importing the TTS class that handles text-to-speech processing
from main.components.TTS.TTS import TTS

class tts_pipeline:
    """
    The `tts_pipeline` class serves as the entry point to tie together TTS configuration 
    and the actual TTS processing logic. It initializes configuration settings and runs the TTS engine.
    """
    
    def __init__(self):
        """
        Initializes the TTS pipeline by fetching configuration and parameter objects 
        from the tts_configration class.
        
        Attributes:
        ----------
        self.config : dict or object
            Configuration settings required by the TTS engine (e.g., model path, language).
        
        self.params : dict or object
            Parameter values used during the TTS process (e.g., pitch, speed, voice type).
        """
        config_obj = tts_configration()
        self.config, self.params = config_obj.tts_configuration()

    def main(self):
        """
        Executes the TTS engine using the pre-loaded configuration and parameters.
        
        It creates a TTS object, then calls the `tts_root()` function with the sample input text.
        
        Args:
        -----
        None
        
        Returns:
        --------
        None
        """
        tts = TTS(self.config, self.params)
        tts.tts_root('hello')  # Example text to convert to speech

# Entry point of the script
if __name__ == '__main__':
    obj = tts_pipeline()
    obj.main()
