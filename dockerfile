FROM tensorflow/tensorflow:2.19.0-gpu

WORKDIR /app

RUN pip3 install --no-cache-dir torch torchvision torchaudio

COPY requirements.txt ./

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .