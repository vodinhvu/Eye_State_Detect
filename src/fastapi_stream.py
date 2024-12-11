import json
from typing import Generator

import cv2
import tensorrt as trt
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from torchvision import transforms
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI()

# Global variables for models
eye_detect_model = None
eye_state_model = None
context = None
eye_state_input_shape = None
output_shape = None
eye_state_logits = None


# Helper function to load TensorRT engine
def load_trt_engine(engine_path: str):
    logger = trt.Logger(trt.Logger.INFO)
    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        try:
            meta_len = int.from_bytes(f.read(4), byteorder='little')
            metadata = json.loads(f.read(meta_len).decode('utf-8'))
            print('Model metadata:', metadata)
        except UnicodeDecodeError:
            f.seek(0)  # If engine lacks metadata
        return runtime.deserialize_cuda_engine(f.read())


# Helper function to preprocess images
def preprocess_image(image: Image.Image, input_shape: tuple):
    transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    resized_image = image.resize(input_shape[2:])
    return transform(resized_image)[None].to('cuda')


# Video streaming generator
def generate_video(source: str, eye_detect_conf: float, eye_state_conf: float) -> Generator[bytes, None, None]:
    global eye_detect_model, eye_state_model, context, eye_state_input_shape, output_shape, eye_state_logits

    # Initialize video source
    video_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise HTTPException(status_code=404, tail="Source not found or cannot be accessed.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detect_results = eye_detect_model.predict(source=frame, stream=True, conf=eye_detect_conf, imgsz=320,
            save=False)
        for detect_result in detect_results:
            eyes_boxes = detect_result.boxes.xyxy.cpu().to(dtype=torch.int).tolist()
            if not eyes_boxes:
                print('No eye detected.')
                continue

            frame_rgb = detect_result.orig_img[:, :, [2, 1, 0]]  # Convert BGR to RGB
            for box in eyes_boxes:
                eye_crop = frame_rgb[box[1]:box[3], box[0]:box[2]]
                eye_image = Image.fromarray(eye_crop)
                eye_tensor = preprocess_image(eye_image, eye_state_input_shape)

                context.execute_v2([int(eye_tensor.data_ptr()), int(eye_state_logits.data_ptr())])
                eye_state_output = eye_state_logits.softmax(dim=-1)[0]
                print('Eye state output:', eye_state_output)

                if eye_state_output[1] > eye_state_conf:
                    print('Eye is closed.')
                    color = (0, 0, 255)  # Red for closed eyes
                else:
                    print('Eye is open.')
                    color = (255, 0, 0)  # Blue for open eyes

                # Draw bounding box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Encode the frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


@app.on_event("startup")
def load_models():
    global eye_detect_model, eye_state_model, context, eye_state_input_shape, output_shape, eye_state_logits

    # Load detection model
    eye_detect_model = YOLO("/training/eyes-detect/yolov11n/yolov11_last.engine", task='detect')

    # Load state model
    eye_state_model = load_trt_engine("/training/eye-state/mobilenet_v2/mobilenet_v2.engine")
    context = eye_state_model.create_execution_context()
    eye_state_input_shape = eye_state_model.get_binding_shape(0)
    output_shape = eye_state_model.get_binding_shape(1)
    eye_state_logits = torch.zeros(*output_shape, device='cuda')

    # Test run
    image_test = torch.rand(1, 3, 320, 320, dtype=torch.float32)
    results = eye_detect_model(image_test, imgsz=320, save=False)
    print("Yolo test:", results)
    image_test = torch.rand(1, 3, 128, 128, dtype=torch.float, device='cuda')
    context.execute_v2([int(image_test.data_ptr()), int(eye_state_logits.data_ptr())])
    print("Eye state test", eye_state_logits)
    image_test = None


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_video(source="0", eye_detect_conf=0.7, eye_state_conf=0.5),
        media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=open("index.html").read())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
