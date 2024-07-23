FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker

COPY resources/dwpose/*.* /dwpose/
COPY resources/*.* /
RUN pip install -r /requirements.txt

##########################################################################################
# SageMaker requirements
##########################################################################################
## install flask
RUN pip install flask
RUN pip install gunicorn

### Install nginx notebook
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV WORKDIR /

ENTRYPOINT ["python", "/serve.py"]
