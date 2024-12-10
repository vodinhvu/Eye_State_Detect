FROM ultralytics/ultralytics:latest-jetson-jetpack4

WORKDIR /eye-state-detect

COPY . .

CMD pip3 list | grep ultralytics
