import onnxruntime
from scipy.io import wavfile
import scipy.signal as sps 
import sounddevice as sd
import numpy as np
from datetime import datetime
from preprocess import Wav2Vec2Processor

input_size = 100000
new_rate = 16000
AUDIO_MAXLEN = input_size
freq = 441

ort_session = onnxruntime.InferenceSession('wav2vec_small.onnx')

def inferrence(sound):
    processor = Wav2Vec2Processor(is_tokenizer=False)
    tokenizer = Wav2Vec2Processor(is_tokenizer=True)
    samples = round(len(sound) * float(new_rate) / freq)
    new_data = sps.resample(np.squeeze(sound,1), samples)
    # new_data = sps.resample(recording, samples)
    new_data = processor(new_data)[None]
    padding = np.zeros((new_data.shape[0], AUDIO_MAXLEN - new_data.shape[1]))
    speech = np.concatenate([new_data, padding], axis=-1)
    speech = speech.astype(np.float32)
    ort_inputs = {"modelInput": speech}
    ort_outs = ort_session.run(None, ort_inputs)
    prediction = np.argmax(ort_outs, axis=-1)
    prediction = tokenizer.decode(prediction.squeeze().tolist())
    return prediction

answer = input("Do you want to speak Y/N :")
while answer=='y':
    # Sampling frequency
    freq = 44100

    # Recording duration
    duration = 5
    print("start record")
    recording = sd.rec(int(duration * freq), 
                       samplerate=freq, channels=1)

    # Record audio for the given number of seconds
    sd.wait()
    print("stop record")
    prediction = inferrence(recording)
    print(prediction)
    answer = input("Do you want to speak Y/N :")
    