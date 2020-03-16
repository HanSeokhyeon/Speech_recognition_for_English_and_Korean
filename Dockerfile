FROM		pytorch/pytorch:0.4.1-cuda9-cudnn7-devel
MAINTAINER	sah0322@naver.com

RUN     apt-get update -y && apt-get -y install parallel sox
RUN		pip install --upgrade pip
RUN		pip install	python-Levenshtein python_speech_features pydub joblib tqdm tensorboardX pandas scipy editdistance tensorflow

WORKDIR		/opt/project
