import argparse
import json

import tensorrt as trt
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


def main(args):
    # Load eye detect model Yolo
    eye_detect_model = YOLO(args.eye_detect_model, task='detect')

    # Load eye state classify model
    logger = trt.Logger(trt.Logger.INFO)
    with open(args.eye_state_model, 'rb') as f, trt.Runtime(logger) as runtime:
        try:
            meta_len = int.from_bytes(f.read(4), byteorder='little')  # read metadata length
            metadata = json.loads(f.read(meta_len).decode('utf-8'))  # read metadata
            print('Model metadata', metadata)
        except UnicodeDecodeError:
            f.seek(0)  # engine file may lack embedded metadata
        eye_stat_model = runtime.deserialize_cuda_engine(f.read())

    context = eye_stat_model.create_execution_context()
    eye_state_input_name = eye_stat_model.get_binding_name(0)
    # print('Eye state input name', eye_state_input_name)
    eye_state_input_shape = eye_stat_model.get_binding_shape(0)
    # print('Eye state input shape', eye_state_input_shape)
    output_name = eye_stat_model.get_binding_name(1)
    # print('Eye state output name', output_name)
    output_shape = eye_stat_model.get_binding_shape(1)
    # print('Eye state output shape', output_shape)
    eye_state_logit = torch.zeros(*output_shape, device='cuda')
    pil_to_tensor = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ])

    # Detect
    detect_results = eye_detect_model.predict(source=args.source, stream=True, conf=args.eye_detect_conf, imgsz=320,
                                              save=False)
    for detect_result in detect_results:
        eyes_box = detect_result.boxes.xyxy.cpu().to(dtype=torch.int).tolist()
        if not eyes_box:
            print('No eye appear')
            continue
        frame_image = detect_result.orig_img[:, :, [2, 1, 0]]
        eyes_zone = [frame_image[box[1]:box[3], box[0]:box[2]] for box in eyes_box]

        for eye in eyes_zone:
            eye_image = Image.fromarray(eye).resize(eye_state_input_shape[2:])
            eye_tensor = pil_to_tensor(eye_image)[None].to('cuda')
            context.execute_v2([int(eye_tensor.data_ptr()), int(eye_state_logit.data_ptr())])
            eye_state_output = eye_state_logit.softmax(dim=-1)[0]
            print('eye_state_output:',
                  eye_state_output)  # if eye_state_output[0] > args.eye_detect_conf:  #     print('Eye is close.')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eye state detect', add_help=True)
    parser.add_argument('--source', default='0', type=str, help='Path to video need process ("0" is use webcam)')
    parser.add_argument('--eye-detect-model', type=str, required=True, help='Path to file model yolo')
    parser.add_argument('--eye-detect-conf', type=float, default=0.7, help='')
    parser.add_argument('--eye-state-model', type=str, required=True, help='Path to model eye')
    parser.add_argument('--eye-state-conf', type=float, default=0.5, help='')
    args = parser.parse_args()
    main(args)
