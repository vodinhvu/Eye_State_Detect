FROM ultralytics/ultralytics:8.3.48-jetson-jetpack4

WORKDIR /eye-state-detect

RUN pip3 install fastapi uvicorn opencv-python-headless

COPY . .

CMD python3 src/main.py
