import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename="live_input.wav", duration=5, fs=22050):
    print("🎙️ Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print(" Recording complete and saved please wait for the predection...")