# Train YOLO Detection Model with TorchVision & Export to TensorRT

This folder contains instructions and relevant documentation links for training a YOLO detection model using TorchVision and exporting it using the provided `export_engine.py`.

---

## Train the YOLO Detection Model

You can train the YOLO detection model using **TorchVision** based on the reference code from PyTorch's official GitHub repository.

- Reference code: [TorchVision GitHub Classification Examples](https://github.com/pytorch/vision/tree/main/references/classification)

---

## Export the Model to TensorRT

Once you have trained your model, export it to TensorRT using the script provided in this directory: `export_engine.py`.  

**Note:** You should run this export on the device where you intend to deploy the TensorRT model.

# Additional Notes
1. **Training Flexibility**:
    You can train the model on any machine with the required PyTorch environment set up.

2. **Export Dependencies**:
Ensure that TensorRT and CUDA are configured on the target deployment environment when running the export_engine.py.






