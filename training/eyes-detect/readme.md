# Ultralytics YOLO Training for Eye Detection

Train the Ultralytics YOLO model (version v11n) for eye detection to ensure optimal performance on NVIDIA Jetson Nano devices. To achieve the best results, we recommend running the training on an NVIDIA GPU server.

For detailed instructions on training YOLOv11n, refer to the official Ultralytics documentation:

- [Training YOLO Documentation](https://docs.ultralytics.com/modes/train/#usage-examples)

---

# Exporting to TensorRT for Deployment

After training, export the YOLOv11n model for eye detection to TensorRT for optimized deployment on the NVIDIA Jetson Nano. Use the official Ultralytics YOLO Docker image to streamline the export process.

- [Export to TensorRT Guide](https://docs.ultralytics.com/guides/nvidia-jetson/#use-tensorrt-on-nvidia-jetson)

**Note:** Ensure the export is performed on the target NVIDIA Jetson Nano device for deployment.

---
