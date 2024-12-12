FROM ultralytics/ultralytics:latest-jetson-jetpack4

WORKDIR /eye-state-detect

RUN pip3 install fastapi uvicorn opencv-python-headless

COPY . .

CMD python3 src/main.py
