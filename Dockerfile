# syntax=docker/dockerfile:1

FROM ubuntu:20.04

RUN apt-get update -y && \
    apt-get install -y python3 python3-pip

COPY ./server_requirements.txt ./server/requirements.txt
COPY ./config.yml ./server/config.yml
COPY ./epoch=49-step=579000.ckpt ./server/epoch=49-step=579000.ckpt
WORKDIR ./server

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install -r requirements.txt

# ENTRYPOINT [ "python3" ]

CMD [ "python3", "app/main.py", "--config_file", "config.yml" ]