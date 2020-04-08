FROM tensorflow/tensorflow:nightly-gpu-py3

RUN apt update && apt install -y wget curl vim git


ADD ./ /project
ADD docker/execute.sh /project/execute.sh

WORKDIR /project

RUN pip install -U pip
RUN pip install -r requirements.txt
RUN pip install awscli