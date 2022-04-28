from scipy.io import wavfile
import scipy.signal as sps 
import sounddevice as sd
import numpy as np
from datetime import datetime
from preprocess import Wav2Vec2Processor
import soundfile as sf
## tacrotron
import torch
from torch import nn
from torch.nn import functional as F
import argparse
import onnxruntime as rt
from inference_trt import init_decoder_inputs
import models
from inference import checkpoint_from_distributed, unwrap_distributed, load_and_setup_model, prepare_input_sequence
from tacotron2_common.utils import to_gpu, get_mask_from_lengths
## Fastapi
from fastapi import FastAPI 
from fastapi import FastAPI, File, Form, UploadFile
from io import BytesIO
import uvicorn

input_size = 100000
new_rate = 16000
AUDIO_MAXLEN = input_size
freq = 44100

# Load model.onnx form directory
ort_session = rt.InferenceSession('model_onnx/wav2vec_small.onnx')
encoder = rt.InferenceSession('model_onnx/encoder.onnx') 
decoder = rt.InferenceSession('model_onnx/decoder_iter.onnx')
postnet = rt.InferenceSession('model_onnx/postnet_2.onnx')
waveglow = rt.InferenceSession('model_onnx/waveglow.onnx')

# Function for change speech to text
def inferrence(sound):
    processor = Wav2Vec2Processor(is_tokenizer=False)
    tokenizer = Wav2Vec2Processor(is_tokenizer=True)
    samples = round(len(sound) * float(new_rate) / freq)
    new_data = sps.resample(np.squeeze(sound,1), samples)
    # new_data = sps.resample(sound, samples)
    new_data = processor(new_data)[None]
    padding = np.zeros((new_data.shape[0], AUDIO_MAXLEN - new_data.shape[1]))
    speech = np.concatenate([new_data, padding], axis=-1)
    speech = speech.astype(np.float32)
    ort_inputs = {"modelInput": speech}
    ort_outs = ort_session.run(None, ort_inputs)
    prediction = np.argmax(ort_outs, axis=-1)
    prediction = tokenizer.decode(prediction.squeeze().tolist())
    return prediction

def text_to_speech(text):
    ## Encoder 
    input_length = [50]
    sequences, lengths = prepare_input_sequence([text],cpu_run=True)
    sequences =  np.int64(sequences.cpu())
    while len(sequences) < input_length[0]:
        sequences = np.append(sequences,0)
    sequences = np.expand_dims(sequences,0)
    memory, processed_memory, lens = encoder.run(["memory","processed_memory","lens"],{"sequences":sequences,"sequence_lengths":input_length})
    memory, processed_memory, lens = torch.tensor(memory),torch.tensor(processed_memory),torch.tensor(lens)

    ## Decoder_iter
    mel_lengths = np.zeros([memory.size(0)], dtype=np.int32)
    not_finished = np.ones([memory.size(0)], dtype=np.int32)
    mel_outputs, gate_outputs, alignments = (np.zeros(1), np.zeros(1), np.zeros(1))


    (decoder_input, attention_hidden, attention_cell, decoder_hidden,
        decoder_cell, attention_weights, attention_weights_cum,
        attention_context, memory, processed_memory,
        mask) = init_decoder_inputs(memory, processed_memory, torch.tensor(input_length))

    (decoder_input, attention_hidden, attention_cell, decoder_hidden,decoder_cell, attention_weights, attention_weights_cum,attention_context, memory, processed_memory,mask)  = decoder_input.numpy(), attention_hidden.numpy(), attention_cell.numpy(), decoder_hidden.numpy(),decoder_cell.numpy(), attention_weights.numpy(), attention_weights_cum.numpy(),attention_context.numpy(),memory.numpy(), processed_memory.numpy(),mask.numpy()

    gate_threshold = 0.6
    max_decoder_steps = 1000
    first_iter = True

    while True:
        decoder_output_name = ["decoder_output", "gate_prediction","out_attention_hidden", "out_attention_cell","out_decoder_hidden", "out_decoder_cell","out_attention_weights", "out_attention_weights_cum","out_attention_context"]
        decoder_input_element = {"decoder_input":decoder_input, "attention_hidden": attention_hidden, "attention_cell":attention_cell, "decoder_hidden":decoder_hidden ,"decoder_cell":decoder_cell, "attention_weights":attention_weights, "attention_weights_cum":attention_weights_cum,"attention_context":attention_context, "memory":memory, "processed_memory":processed_memory,"mask":mask}

        (mel_output, gate_output,
                    attention_hidden, attention_cell,
                    decoder_hidden, decoder_cell,
                    attention_weights, attention_weights_cum,
                    attention_context) = decoder.run(decoder_output_name , decoder_input_element)

        if first_iter:
            mel_outputs = np.expand_dims(mel_output, 2)
            gate_outputs = np.expand_dims(gate_output, 2)
            alignments = np.expand_dims(attention_weights, 2)
            first_iter = False
        else:
            mel_outputs = np.concatenate((mel_outputs, np.expand_dims(mel_output, 2)), 2)
            gate_outputs = np.concatenate((gate_outputs, np.expand_dims(gate_output, 2)), 2)
            alignments =np.concatenate((alignments, np.expand_dims(attention_weights, 2)), 2)
            
        dec = torch.le(torch.sigmoid(torch.Tensor(gate_output)), gate_threshold).to(torch.int32).squeeze(1)
        not_finished = not_finished*dec.numpy()
        mel_lengths += not_finished
        
        if np.sum(not_finished) == 0:
                print("Stopping after ",mel_outputs.size," decoder steps")
                break
        if mel_outputs.size == max_decoder_steps:
            print("Warning! Reached max decoder steps")
            break
            
        decoder_input = mel_output
        
    #postnet
    mel_outputs_postnet = postnet.run(['mel_outputs_postnet'],{'mel_outputs':mel_outputs})
    stride = 256 # value from waveglow upsample
    n_group = 8
    z_size2 = (mel_outputs_postnet[0].shape[2]*stride)//n_group
    z = np.random.randn(1, n_group, z_size2)
    audio = waveglow.run(['audio'],{'mel':mel_outputs_postnet[0].astype('float32'),'z':z.astype('float32')})
    audio_numpy = np.squeeze(audio[0])
    audio = None
    return audio_numpy

app = FastAPI()

@app.get("/")
def Home():
    return {"Voice recognition app"}

@app.post("/prediction/")
async def speech_reg(audio_upload: UploadFile = File(...)):
    audio_data = await audio_upload.read()
    buffer = BytesIO(audio_data)
    data, fs = sf.read(buffer, dtype='float32')
    data_numpy = np.expand_dims(data,axis=1)
    text = inferrence(data_numpy)
    audio_numpy = text_to_speech(text)
    return {f"{audio_numpy}"}