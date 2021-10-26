FROM lileee/ubuntu-16.04-cuda-9.0-python-3.5-pytorch:latest

ADD 3dmv/ 3dmv/
ADD prepare_data prepare_data/
ADD scripts scripts/

