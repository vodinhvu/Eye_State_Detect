# Image Classification with TorchVision & Exporting to TensorRT

This folder contains instructions for training an **image classification model** using **TorchVision** and exporting it to **TensorRT** for deployment on an **NVIDIA Jetson Nano**.

---

## 1. Train the Model

Use **TorchVision** to train the image classification model. For efficiency, it's recommended to train a **MobileNetV2** model, which is lightweight and suitable for deployment on resource-constrained devices like the **NVIDIA Jetson Nano**.

- Reference: [TorchVision GitHub Classification Examples](https://github.com/pytorch/vision/tree/main/references/classification)

**Training Environment:**  
Train on an **NVIDIA GPU server** with the latest **TorchVision** version. However, ensure compatibility with the export environment, as the latest version of **TorchVision** may not be supported for export to **TensorRT**.

---

## 2. Export to TensorRT

After training, use the `export_engine.py` script to export the model to **TensorRT**. This process must be done on the same device where the model will be deployed (i.e., the **NVIDIA Jetson Nano**).

**Deployment Environment:**  
Because the **Jetson Nano** has limited resources compared to an **NVIDIA GPU server**, exporting a smaller model like **MobileNetV2** ensures optimal performance during deployment.

**Compatibility Note:**  
Ensure that both the **training** and **export** environments use compatible versions of **TorchVision** for smooth export to **TensorRT**.

---

## Additional Notes

- **Training:** Performed on an **NVIDIA GPU server** with the latest **TorchVision** version.
- **Export:** Must be done on the **NVIDIA Jetson Nano** with **TensorRT** and **CUDA** configured properly.
- **Performance:** For the **Jetson Nano**, choose a lightweight model like **MobileNetV2** to ensure optimal performance during deployment.
