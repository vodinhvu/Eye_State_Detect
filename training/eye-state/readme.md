# Image Classification with TorchVision & TensorRT Export

This guide explains how to train an **image classification model** using **TorchVision** and export it to **TensorRT** for deployment on an **NVIDIA Jetson Nano**.

---

## 1. Train the Model

Train an image classification model using **TorchVision**. We recommend **MobileNetV2**, a lightweight model ideal for resource-constrained devices like the **Jetson Nano**.

- **Environment:** Train on an **NVIDIA GPU server** with **TorchVision**. Ensure the version is compatible with **TensorRT** for export.  
- **Reference:** [TorchVision Classification Examples](https://github.com/pytorch/vision/tree/main/references/classification)

After training, you will obtain a **model checkpoint**. This checkpoint contains the model's `state_dict`, which will be required for the export process.

---

## 2. Export to TensorRT

After training the model, export it for deployment using the provided `export_engine.py` script. This script supports the conversion of a **PyTorch checkpoint** to **ONNX** and then to **TensorRT**.

### Prerequisites
- **Jetson Nano Environment:** Ensure TensorRT and CUDA are properly configured.
- **TorchVision Model Compatibility:** Use models like MobileNetV2 for optimal performance.

### Usage Instructions
1. **Load the Script:**
   Use `export_engine.py` to process the trained model checkpoint.
   
2. **Run the Export Process:**
   ```bash
   python export_engine.py --model MobileNetV2 --checkpoint path/to/checkpoint.pt --input-size 128
   ```

3. **Expected Outputs:**
   - **ONNX Model:** Generated if not already present (e.g., `checkpoint.onnx`).
   - **TensorRT Engine:** Exported for inference (e.g., `checkpoint.engine`).

4. **Script Workflow:**
   - **Step 1:** Load the model checkpoint and verify its parameters.
   - **Step 2:** Export the PyTorch model to ONNX format.
   - **Step 3:** Convert the ONNX model to TensorRT engine format.
   - **Step 4:** Test the TensorRT engine with a sample input.

5. **Additional Notes:**
   - Metadata is embedded in the exported ONNX and TensorRT models.
   - Ensure consistent TorchVision and TensorRT versions to avoid compatibility issues.

