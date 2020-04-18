FROM tensorflow/tensorflow:1.15.2-gpu

RUN apt update && apt install -y wget curl vim git groff
RUN pip install -U pip
RUN pip install tensor2tensor


ADD ./ /project
ADD docker/execute.sh /project/execute.sh

WORKDIR /project

