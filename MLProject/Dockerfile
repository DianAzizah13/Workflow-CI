FROM continuumio/miniconda3

COPY . /app
WORKDIR /app

RUN conda env create -f conda.yaml

ENV PATH /opt/conda/envs/crop-env/bin:$PATH

CMD ["python", "modelling.py"]
