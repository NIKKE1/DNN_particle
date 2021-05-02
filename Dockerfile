FROM nvcr.io/nvidia/tensorflow:21.04-tf2-py3

WORKDIR /basicdnn/

RUN pip3 install –upgrade pip && pip3 install -r requirements.txt
RUN python dataloader.py

cmd ["python","basicdnn.py"]
