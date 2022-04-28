from fastapi import FastAPI 
from fastapi import FastAPI, File, Form, UploadFile
import uvicorn
from io import BytesIO
import soundfile as sf
import sounddevice as sd
import onnxruntime as rt
from preprocess import Wav2Vec2Processor

ort_session = rt.InferenceSession('model_onnx/wav2vec_small.onnx')

# def inferrence(sound):
#     processor = Wav2Vec2Processor(is_tokenizer=False)
#     tokenizer = Wav2Vec2Processor(is_tokenizer=True)
#     samples = round(len(sound) * float(new_rate) / freq)
#     new_data = sps.resample(np.squeeze(sound,1), samples)
#     # new_data = sps.resample(recording, samples)
#     new_data = processor(new_data)[None]
#     padding = np.zeros((new_data.shape[0], AUDIO_MAXLEN - new_data.shape[1]))
#     speech = np.concatenate([new_data, padding], axis=-1)
#     speech = speech.astype(np.float32)
#     ort_inputs = {"modelInput": speech}
#     ort_outs = ort_session.run(None, ort_inputs)
#     prediction = np.argmax(ort_outs, axis=-1)
#     prediction = tokenizer.decode(prediction.squeeze().tolist())
#     return prediction

app = FastAPI()

@app.get("/")
def Home():
    return {"Voice recognition app"}

@app.post("/prediction/")
async def speech_reg(audio_upload: UploadFile = File(...)):
    audio_data = await audio_upload.read()
    buffer = BytesIO(audio_data)
    data, fs = sf.read(buffer, dtype='float32')
    prediction = inferrence(data)
    return {"text is : ":prediction}

# @app.get("/cal/{a}/{b}")
# def calculus(a:int,b:int):
#     result = a+b
#     return {"result":result}



