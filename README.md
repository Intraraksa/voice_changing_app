# tacotron2_onnx
This repository is created for demo application. Capable deploy to device

This repository included 4 onnx model by downloading from [here](https://drive.google.com/drive/folders/1E4aKAM9plOy9lOtl38vRPJKYKnojLxYH?usp=sharing)

## for Docker command

You can run application by jupyterlab by following step
1. build dockerfile
2. run docker
3. copy link address to your browser
4. run notebook tts_botnoi.ipynb

**docker build:** 
sudo docker build --rm -t tts .

**docker run:**
sudo docker run -rm --name tts -p 8888:8888  tts:latest





