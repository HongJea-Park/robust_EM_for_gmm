FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install sudo
RUN sudo apt update -y
RUN sudo apt upgrade -y
RUN sudo apt-get install -y \
    python3-venv
RUN sudo ln -s /usr/bin/python3.6 /usr/bin/python

WORKDIR /app