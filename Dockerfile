FROM python:3.9 AS labelatorio-serving-base
LABEL author=blip-solutions version=0.1

RUN pip3 install torch  --extra-index-url https://download.pytorch.org/whl/cpu

FROM labelatorio-serving-base AS labelatorio-serving-torch


WORKDIR /app
COPY requirements.txt  .
RUN pip3 install -r requirements.txt
FROM labelatorio-serving-torch AS labelatorio-serving-requirements
RUN pip3 install labelatorio==0.3.4
RUN pip3 install openai

COPY . /app
EXPOSE 80

ARG build_version
ENV BUILD_VERSION=$build_version

# If running alone (when you dont want /api prefix) remove "--root-path" and "/api" from below command  
ENTRYPOINT ["uvicorn", "--host", "0.0.0.0", "--port", "80", "main:app"]
#CMD exec gunicorn --bind :80--workers 1 --threads 8 --timeout 0 main:app

