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
eye_detect_model = None  # YOLO-based eye detection model
eye_state_model = None  # TensorRT-based eye state classification model
context = None  # TensorRT execution context
eye_state_input_shape = None  # Input shape for the eye state model
output_shape = None  # Output shape for the eye state model
eye_state_logits = None  # Tensor to hold model output


def load_trt_engine(engine_path: str):
    """
    Load a TensorRT engine from the specified file path.

    Args:
        engine_path (str): Path to the TensorRT engine file.

    Returns:
        trt.ICudaEngine: The deserialized TensorRT engine.
    """
    logger = trt.Logger(trt.Logger.INFO)
    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        try:
            # Read metadata from engine file
            meta_len = int.from_bytes(f.read(4), byteorder='little')
            metadata = json.loads(f.read(meta_len).decode('utf-8'))
            print('Model metadata:', metadata)
        except UnicodeDecodeError:
            f.seek(0)  # Reset file pointer if metadata is unavailable
        return runtime.deserialize_cuda_engine(f.read())


def preprocess_image(image: Image.Image, input_shape: tuple):
    """
    Preprocess an image for the eye state model.

    Args:
        image (PIL.Image.Image): Input image.
        input_shape (tuple): Expected input shape of the model.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    resized_image = image.resize(input_shape[2:])
    return transform(resized_image)[None].to('cuda')


def generate_video(source: str, eye_detect_conf: float, eye_state_conf: float) -> Generator[bytes, None, None]:
    """
    Stream video frames with eye detection and state classification.

    Args:
        source (str): Video source (camera index or file path).
        eye_detect_conf (float): Confidence threshold for eye detection.
        eye_state_conf (float): Confidence threshold for closed eye classification.

    Yields:
        bytes: Encoded video frame.
    """
    global eye_detect_model, eye_state_model, context, eye_state_input_shape, output_shape, eye_state_logits

    # Initialize video source
    video_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise HTTPException(status_code=404, detail="Source not found or cannot be accessed.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detect_results = eye_detect_model.predict(source=frame, stream=True, conf=eye_detect_conf, imgsz=320,
                                                  save=False)

        for detect_result in detect_results:
            eyes_boxes = detect_result.boxes.xyxy.cpu().to(dtype=torch.int).tolist()

            if not eyes_boxes:
                print("Warning: No eye detected in the current frame.")
                cv2.putText(frame, "Warning: No eye detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue

            for box in eyes_boxes:
                # Crop and preprocess eye image
                eye_crop = frame[box[1]:box[3], box[0]:box[2]]
                eye_image = Image.fromarray(eye_crop)
                eye_tensor = preprocess_image(eye_image, eye_state_input_shape)

                # Perform inference
                context.execute_v2([int(eye_tensor.data_ptr()), int(eye_state_logits.data_ptr())])
                eye_state_output = eye_state_logits.softmax(dim=-1)[0]

                if eye_state_output[1] > eye_state_conf:
                    color = (0, 0, 255)  # Red for closed eyes
                    cv2.putText(frame, "Warning: Eye is closed", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 2)
                else:
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
    """
    Load the eye detection and state classification models during application startup.
    """
    global eye_detect_model, eye_state_model, context, eye_state_input_shape, output_shape, eye_state_logits

    # Load YOLO-based eye detection model
    eye_detect_model = YOLO("training/eyes-detect/yolov11n/yolov11_last.engine", task='detect')

    # Load TensorRT-based eye state classification model
    eye_state_model = load_trt_engine("training/eye-state/mobilenet_v2/mobilenet_v2.engine")
    context = eye_state_model.create_execution_context()
    eye_state_input_shape = eye_state_model.get_binding_shape(0)
    output_shape = eye_state_model.get_binding_shape(1)
    eye_state_logits = torch.zeros(*output_shape, device='cuda')

    # Test model inference
    image_test = torch.rand(1, 3, 320, 320, dtype=torch.float32)
    results = eye_detect_model(image_test, imgsz=320, save=False)
    print("YOLO test result:", results)

    image_test = torch.rand(1, 3, 128, 128, dtype=torch.float, device='cuda')
    context.execute_v2([int(image_test.data_ptr()), int(eye_state_logits.data_ptr())])
    print("Eye state test result:", eye_state_logits)


@app.get("/video_feed")
def video_feed():
    """
    Serve the video feed with eye detection and state classification.

    Returns:
        StreamingResponse: Video stream response.
    """
    return StreamingResponse(generate_video(source="0", eye_detect_conf=0.7, eye_state_conf=0.5),
        media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Serve the home page.

    Returns:
        HTMLResponse: Home page content.
    """
    return HTMLResponse(content=open("src/index.html").read())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
