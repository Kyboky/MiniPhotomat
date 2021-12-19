# syntax=docker/dockerfile:1

FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
RUN pip3 install opencv-python-headless

COPY main.py main.py
COPY model.h5 model.h5
COPY model.json model.json
COPY index.html index.html

CMD [ "python3", "-m" , "main"]