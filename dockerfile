FROM python:3

ENV PYTHONBUFFERED=1
WORKDIR /app

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install  --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY .. .

CMD ["python","fall_detection.py"]