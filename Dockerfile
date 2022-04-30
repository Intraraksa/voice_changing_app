from pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
# from ubuntu

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install git -y

RUN apt-get install libportaudio2 libasound-dev libsndfile1

RUN pip3 install -r requirements.txt


EXPOSE 8888

CMD ["uvicorn","service_voice_change:app","--host","0.0.0.0","--port","8000"]
