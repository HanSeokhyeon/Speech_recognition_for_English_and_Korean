FROM		pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
MAINTAINER	sah0322@naver.com

RUN		pip install --upgrade pip
RUN		pip install	python-Levenshtein python_speech_features pydub joblib tqdm tensorboardX pandas scipy editdistance tensorflow

WORKDIR		/opt/project
