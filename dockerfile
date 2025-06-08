FROM tensorflow/tensorflow:2.19.0-gpu

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .