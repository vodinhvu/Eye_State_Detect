import argparse
import json
import pathlib
from typing import Union

import onnx
import onnxruntime as ort
import tensorrt as trt
import torch
import torchvision

torch.set_grad_enabled(False)

def load_model(model: str, file: Union[str, pathlib.Path], device: str = 'cuda'):
    """
    Load a PyTorch model from a checkpoint file and prepare it for inference.

    Args:
        model (str): The name of the model architecture (e.g., 'mobilenet_v2').
        file (Union[str, pathlib.Path]): Path to the PyTorch checkpoint file.
        device (str): Device to load the model onto ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    info = torch.load(file, map_location=torch.device('cpu'))
    model = torchvision.models.__dict__[model](num_classes=2, pretrained=False)
    model.load_state_dict(info['model'])
    model.to(device=device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    model.eval()
    print('Pytorch total params:', pytorch_total_params)
    return model

def export_onnx(model: torch.nn.Module, input_sample: torch.Tensor, save_file: Union[str, pathlib.Path]):
    """
    Export a PyTorch model to ONNX format and attach metadata.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        input_sample (torch.Tensor): A sample input tensor for the model.
        save_file (Union[str, pathlib.Path]): Path to save the ONNX model.

    Returns:
        None
    """
    # Export to ONNX
    torch.onnx.export(model.cpu(), input_sample.cpu(), save_file, verbose=False, opset_version=11,
                      input_names=['images'], output_names=['output0'], dynamic_axes=None)
    # Add metadata
    model_onnx = onnx.load(save_file)  # Load ONNX model
    metadata = {'model': 'MobileNetV2', 'author': 'vdvu'}
    for k, v in metadata.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, save_file)

def onnx_to_engine(input_file: Union[str, pathlib.Path], output_file: Union[str, pathlib.Path], half: bool = True):
    """
    Convert an ONNX model to a TensorRT engine.

    Args:
        input_file (Union[str, pathlib.Path]): Path to the ONNX model file.
        output_file (Union[str, pathlib.Path]): Path to save the TensorRT engine file.
        half (bool): Whether to enable FP16 precision for inference.

    Returns:
        None
    """
    print(f'Starting export with TensorRT {trt.__version__}...')
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)

    # Parse ONNX file
    trt_parser = trt.OnnxParser(network, logger)
    if not trt_parser.parse_from_file(str(input_file)):
        raise RuntimeError(f'Failed to load ONNX file: {input_file}')

    # Log network input/output details
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'Input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'Output "{out.name}" with shape{out.shape} {out.dtype}')

    if half:
        config.set_flag(trt.BuilderFlag.FP16)

    # Build and save engine
    build = builder.build_engine
    with build(network, config) as engine, open(output_file, 'wb') as t:
        meta = json.dumps({'model': 'MobileNetV2', 'author': 'vdvu'})
        t.write(len(meta).to_bytes(4, byteorder='little', signed=True))
        t.write(meta.encode())
        t.write(engine.serialize())

def run_engine(file: Union[str, pathlib.Path], input_sample: torch.Tensor):
    """
    Run inference using a TensorRT engine file.

    Args:
        file (Union[str, pathlib.Path]): Path to the TensorRT engine file.
        input_sample (torch.Tensor): Input tensor for inference.

    Returns:
        torch.Tensor: The output tensor from the engine.
    """
    logger = trt.Logger(trt.Logger.INFO)

    with open(file, 'rb') as f, trt.Runtime(logger) as runtime:
        try:
            meta_len = int.from_bytes(f.read(4), byteorder='little')  # Read metadata length
            metadata = json.loads(f.read(meta_len).decode('utf-8'))  # Read metadata
            print('Model metadata:', metadata)
        except UnicodeDecodeError:
            f.seek(0)  # Engine file may lack embedded metadata
        model = runtime.deserialize_cuda_engine(f.read())

    context = model.create_execution_context()
    input_name = model.get_binding_name(0)
    print('TensorRT input name:', input_name)
    input_shape = model.get_binding_shape(0)
    print('TensorRT input shape:', input_shape)
    if input_sample.shape != input_shape:
        print(f'Input sample shape {input_sample.shape}, but engine requires {input_shape}')

    output_name = model.get_binding_name(1)
    print('TensorRT output name:', output_name)
    output_shape = model.get_binding_shape(1)
    print('TensorRT output shape:', output_shape)

    input_sample = input_sample.to(device='cuda')
    output = torch.zeros(*output_shape, device='cuda')
    context.execute_v2([int(input_sample.data_ptr()), int(output.data_ptr())])

    return output

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Eye state detection', add_help=True)
    parser.add_argument('--model', type=str, required=True, help='Name of the model architecture.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the PyTorch model checkpoint.')
    parser.add_argument('--input-size', type=int, default=128, help='Size of the input image.')
    args = parser.parse_args()

    # Validate checkpoint file
    model_checkpoint = pathlib.Path(args.checkpoint)
    if not model_checkpoint.exists():
        raise FileNotFoundError('Error: Input model file does not exist.')

    # Load PyTorch model
    torch_model = load_model(model=args.model, file=model_checkpoint, device='cuda')
    input_tensor = torch.randn(1, 3, args.input_size, args.input_size, device='cuda')
    print('Random input shape:', input_tensor.shape)
    pytorch_output = torch_model(input_tensor)
    print('PyTorch output:', pytorch_output)

    # Export to ONNX
    onnx_file = model_checkpoint.with_suffix('.onnx')
    if not onnx_file.exists():
        export_onnx(model=torch_model, input_sample=input_tensor, save_file=onnx_file)
    else:
        print(f'Using existing ONNX model file: {onnx_file}')

    ort_sess = ort.InferenceSession(str(onnx_file))
    onnx_output = ort_sess.run(None, {'images': input_tensor.cpu().numpy()})
    print('ONNX output:', onnx_output)

    # Convert ONNX to TensorRT engine
    engine_file = model_checkpoint.with_suffix('.engine')
    if not engine_file.exists():
        onnx_to_engine(input_file=onnx_file, output_file=engine_file)
    else:
        print(f'Using existing TensorRT engine file: {engine_file}')

    # Test TensorRT engine
    engine_output = run_engine(file=engine_file, input_sample=input_tensor)
    print('TensorRT engine output:', engine_output)