from main.components.speech_recognition.config.speech_recog_config import speech_recog_config
from faster_whisper import WhisperModel
import torch, wave, io, os
import speech_recognition as sr
from scipy.io.wavfile import write


class speech_recog_compo:
    def __init__(self, config, parmas):
        self.config = config
        self.params = parmas
        self.is_recording = False



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
        print("🎙️ Listening... Speak now. Recording will stop automatically when you're silent.\n(Press Ctrl+C to stop manually)\n")

        recog = sr.Recognizer()
        mic = sr.Microphone(sample_rate=self.params['sample_rate'])
        all_audio = []

        # try:
        with mic as source:
            recog.adjust_for_ambient_noise(source)
            while True:
                print("🎧 Listening for next phrase...")
                try:
                    audio = recog.listen(source, timeout=2, phrase_time_limit=10)
                    all_audio.append(audio.get_wav_data())
                    print("✅ Phrase captured.")
                except sr.WaitTimeoutError:
                    print("⏹️ Detected silence. Ending recording.")
                    break
        # except KeyboardInterrupt:
        #     print("\n🛑 Manual stop triggered.")

        if not all_audio:
            print("⚠️ No audio was recorded.")
            return

        wav_file = "recording.wav"
        self._combine_audio_data(all_audio, wav_file)
        print("✅ Recording saved to:", wav_file)

        text = self._transcribe_audio(wav_file)
        return text
    
    def _combine_audio_data(self, audio_data_chunks, output_filename):
        """Combine multiple audio chunks into a single WAV file"""
        if not audio_data_chunks:
            return
        
        # Get parameters from the first chunk
        with wave.open(io.BytesIO(audio_data_chunks[0]), 'rb') as first_chunk:
            params = first_chunk.getparams()
        
        # Write all chunks to a single file
        with wave.open(output_filename, 'wb') as output_file:
            output_file.setparams(params)

            for chunk in audio_data_chunks:
                with wave.open(io.BytesIO(chunk), 'rb') as chunk_file:
                    output_file.writeframes(chunk_file.readframes(chunk_file.getnframes()))
    
    def _transcribe_audio(self, wav_file):
        """Transcribe the recorded audio using Whisper"""
        model = WhisperModel(self.config['model'], compute_type=self.config['compute_type'])
        
        segments, info = model.transcribe(wav_file, log_progress=True, beam_size=5)
        print("🌐 Detected Language:", info.language)
        print("📝 Transcription:")
        full_text = ""
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        full_text += segment.text.strip() + " "
        return full_text.strip() if full_text else None

if __name__ == '__main__':
    print(" 🎙️🎙️ Initializing Speech Recognition Component...")
    print("✔️ Loading configuration...")

    obj = speech_recog_config()
    config_params = obj.speech_recog_module()
    
    f = speech_recog_compo(config_params, config_params)
    f.download_model()
    d = f.record_and_transcribe_audio()
    print("Final Transcription:", d)
    print(type(d))