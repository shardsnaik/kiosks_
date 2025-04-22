from main.components.speech_recognition.config.speech_recog_config import speech_recog_config
from faster_whisper import WhisperModel
import torch
import sounddevice
import speech_recognition as sr
from scipy.io.wavfile import write
import os

class speech_recog_compo:
    def __init__(self, config, parmas):
        # obj = speech_recog()
        # self.configs = obj.speech_recog_module()
        # self.model_path = self.configs['model_path']
        #  print(self.configs)
        self.config = config
        self.params = parmas


    def download_model(self):
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"  # <-- ADD THIS LINE
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        model_size = self.config['model']
        model = WhisperModel(model_size, download_root= self.config['model_path'], device=device, compute_type="float32")
        print('model succefully downloaded', model)

    
    def record_and_transcribe_audio(self):
        print('Recording audio...')

        # audio = sounddevice.rec(int(self.params['duration'] * self.params['sample_rate']), samplerate=self.params['sample_rate'], channels=1, dtype='int16')

        
        # 🎙️ recording usign speech_recognition model
        print("🎙️ Listening... Speak now (will stop automatically when you're silent)")

        recog = sr.Recognizer()
        mic = sr.Microphone(sample_rate=self.params['sample_rate'])
        all_audio =[]
        with mic as source:
            recog.adjust_for_ambient_noise(source)
            audio = recog.listen(source, 
            timeout=5, phrase_time_limit=None)

        # sounddevice.wait()
        # write('recording.wav', self.params['sample_rate'], audio)
        # Save to WAV
        wav_file = "recording.wav"
        with open(wav_file, "wb") as f:
            f.write(audio.get_wav_data())
    

        print("✅ Recording saved to", "recording.wav")
        model = WhisperModel(self.config['model'], compute_type=self.config['compute_type'])  

        segments, info = model.transcribe("recording.wav",log_progress=True, beam_size=5)
        print("🌐 Detected Language:", info.language)
        print("📝 Transcription:")
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    

obj = speech_recog_config()
config_params = obj.speech_recog_module()

f = speech_recog_compo(config_params, config_params)
f.download_model()
f.record_and_transcribe_audio()