FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# set bash as current shell
#RUN chsh -s /bin/bash
#SHELL ["/bin/bash", "-c"]

# install anaconda
RUN apt-get update -y
RUN apt-get install python3 -y   
RUN apt-get install python3-pip -y

# setup conda virtual environment
#COPY ./requirements.txt /tmp/requirements.txt
RUN python3  -m pip install --upgrade pip
COPY requirements.txt /tmp/
RUN python3 -m pip install -r /tmp/requirements.txt



ADD scripts /

CMD [ "python3", "download.py"]
