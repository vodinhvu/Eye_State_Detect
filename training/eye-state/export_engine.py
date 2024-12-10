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


def load_model(file: Union[str, pathlib.Path], device: str = 'cuda'):
    info = torch.load(file, map_location=torch.device('cpu'))
    model = torchvision.models.MobileNetV2(num_classes=2)
    model.load_state_dict(info['model'])
    model.to(device=device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Pytorch total params:', pytorch_total_params)
    return model


def export_onnx(model: torch.nn.Module, input_sample: torch.Tensor, save_file: Union[str, pathlib.Path]):
    # Export to Onnx
    torch.onnx.export(model.cpu(), input_sample.cpu(), save_file, verbose=True, opset_version=14,
                      # do_constant_folding=True,
                      # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
                      input_names=['images'], output_names=['output0'], dynamic_axes=None, )
    # Add meta data
    model_onnx = onnx.load(save_file)  # load onnx model
    metadata = {'model': 'MobileNetV2', 'author': 'vdvu'}
    # Metadata
    for k, v in metadata.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, save_file)


def onnx_to_engine(input_file: Union[str, pathlib.Path], output_file: Union[str, pathlib.Path], half: bool = True):
    print(f'Starting export with TensorRT {trt.__version__}...')
    logger = trt.Logger(trt.Logger.INFO)
    # Engine builder
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    # config.max_workspace_size = 2
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)

    # Read ONNX file
    trt_parser = trt.OnnxParser(network, logger)
    if not trt_parser.parse_from_file(str(input_file)):
        raise RuntimeError(f'failed to load ONNX file: {input_file}')

    # Network inputs
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'output "{out.name}" with shape{out.shape} {out.dtype}')

    if half:
        config.set_flag(trt.BuilderFlag.FP16)

    # Write file
    build = builder.build_engine
    with build(network, config) as engine, open(output_file, 'wb') as t:
        # Metadata
        meta = json.dumps({'model': 'MobileNetV2', 'author': 'vdvu'})
        t.write(len(meta).to_bytes(4, byteorder='little', signed=True))
        t.write(meta.encode())
        # Model
        t.write(engine.serialize())


def run_engine(file: Union[str, pathlib.Path], input_sample: torch.Tensor):
    logger = trt.Logger(trt.Logger.INFO)

    with open(file, 'rb') as f, trt.Runtime(logger) as runtime:
        try:
            meta_len = int.from_bytes(f.read(4), byteorder='little')  # read metadata length
            metadata = json.loads(f.read(meta_len).decode('utf-8'))  # read metadata
            print('Model metadata', metadata)
        except UnicodeDecodeError:
            f.seek(0)  # engine file may lack embedded metadata
        model = runtime.deserialize_cuda_engine(f.read())

    context = model.create_execution_context()
    input_name = model.get_binding_name(0)
    print('Tensorrt input name', input_name)
    input_shape = model.get_binding_shape(0)
    print('Tensorrt input shape', input_shape)
    if input_sample.shape != input_shape:
        print(f'Input sample shape {input_sample.shape}, but engine require {input_shape}')

    output_name = model.get_binding_name(1)
    print('Tensorrt output name', output_name)
    output_shape = model.get_binding_shape(1)
    print('Tensorrt output shape', output_shape)

    input_sample = input_sample.to(device='cuda')
    output = torch.zeros(*output_shape, device='cuda')
    context.execute_v2([int(input_sample.data_ptr()), int(output.data_ptr())])

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eye state detect', add_help=True)
    parser.add_argument('--model', type=str, required=True, help='Path to pytorch model checkpoint.')
    parser.add_argument('--input-size', type=int, default=128, help='Size of input image.')
    args = parser.parse_args()

    model_file = pathlib.Path(args.model)
    if not model_file.exists():
        raise 'Error input model file.'

    # torch model
    torch_model = load_model(file=model_file, device='cuda')
    input_tensor = torch.randn(1, 3, args.input_size, args.input_size, device='cuda')
    print('Random input shape:', input_tensor.shape)
    pytorch_output = torch_model(input_tensor)
    print("pytorch output:", pytorch_output)

    # torch to onnx
    onnx_file = model_file.with_suffix('.onnx')
    export_onnx(model=torch_model, input_sample=input_tensor, save_file=onnx_file)
    ort_sess = ort.InferenceSession(str(onnx_file))
    onnx_output = ort_sess.run(None, {'images': input_tensor.cpu().numpy()})
    print('Onnx output:', onnx_output)

    # onnx to tensorrt engine
    engine_file = model_file.with_suffix('.engine')
    onnx_to_engine(input_file=onnx_file, output_file=engine_file)

    # test engine
    engine_output = run_engine(file=engine_file, input_sample=input_tensor)
    print('tensorrt engine output:', engine_output)
