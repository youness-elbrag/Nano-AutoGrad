FROM python:3.9-slim

COPY ./app /app
COPY requirements.txt /app/requirements.txt

WORKDIR /app

# RUN mkdir predictedSamples

RUN apt-get update && \
    if [ $DEV = "true" ]; \
        then pip3 install -r requirements.txt && \
    rm -rf /app/requirements.txt ; \
    fi && \ 
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip



HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health