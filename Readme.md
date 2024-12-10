
# **Eye State Detection**

A compact, real-time system to detect whether a person’s eyes are open or closed. Designed for NVIDIA Jetson Nano, this project uses YOLO for eye detection and a PyTorch classifier for state classification.

---

## **Features**
- **Real-time Detection**: Detect and classify eye states from video feeds.  
- **Pretrained Models**: Includes YOLOv11 and MobileNetV2 for efficient performance.  
- **End-to-End Training Pipeline**: Train detection and classification models with ease.  

---

## **Use Case**
This project is designed for **driver drowsiness detection**. By running the eye state detection system on the **NVIDIA Jetson Nano Kit**, the system monitors the driver’s eyes in real time. If the driver’s eyes are detected as closed for a prolonged period, indicating that they may have fallen asleep, the system can trigger an alarm to alert the driver and prevent potential accidents.

While this setup is compact, efficient, and optimized for the Jetson Nano, it is still under development and requires further optimization for real-world deployment. Additional work is needed to improve the system’s accuracy, responsiveness, and robustness to ensure it is ready for production use in vehicles.

---

## **Requirements**

### **Hardware**
- NVIDIA Jetson Nano  
- Camera (USB or Raspberry Pi)

### **Software**
- **JetPack SDK 4.6**  
- **Python 3.8** (Ensure Python 3.8 is installed on Jetson Nano)

---

## **Project Structure**
```plaintext
├── datasets                   # Datasets for detection and classification
│   ├── eyes-detect-dataset    # Dataset for eye detection
│   └── eyes-state-dataset     # Dataset for eye state classification
├── Dockerfile                 # Docker configuration
├── Readme.md                  # Documentation
├── requirements.txt           # Python dependencies
├── src                        # Source code for the application
│   └── main.py
├── training                   # Model training and exported models
│   ├── eyes-detect            # Eye detection model training folder
│   └── eye-state              # Eye state classification model training folder
```

---

## **Setup**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vodinhvu/Eye_State_Detect.git
   cd eye-state-detection
   ```

2. **Install dependencies**:  
   You can either:
   - **Run with host Python 3.8**:
     Install the required dependencies using `pip`:
     ```bash
     pip install -r requirements.txt
     ```
   
   - **Use Docker** (recommended for a consistent environment):
     Build the Docker image and run the container:
     ```bash
     docker build -t eye-state-detection .
     docker run --rm -it --runtime nvidia eye-state-detection /bin/bash
     ```

---

## **Training**

### **Prepare Datasets**

Before training, ensure the datasets are set up:
- **Eye Detection Dataset**: For instructions on how to download and set up the detection dataset, refer to [datasets/eyes-detect-dataset/readme.md](datasets/eyes-detect-dataset/readme.md).
- **Eye State Classification Dataset**: For instructions on how to download and set up the eye state classification dataset, refer to [datasets/eyes-state-dataset/readme.md](datasets/eyes-state-dataset/readme.md).

### **Training Instructions**

For detailed instructions on how to train each model, refer to the `README.md` files inside the respective training directories:

- **Detection Model**: [training/eyes-detect/readme.md](training/eyes-detect/readme.md)
- **Eye State Classifier**: [training/eye-state/readme.md](training/eye-state/readme.md)

The training process is designed to be performed on a server with an NVIDIA GPU, not on the Jetson Nano.

---

## **Inference**

The inference is designed to run on NVIDIA Jetson Nano using the pretrained models. To run the application, simply execute:
```bash
python src/main.py
```

---

## **Acknowledgments**

This project utilizes NVIDIA Jetson Nano, YOLOv11, and MobileNetV2 for efficient AI-powered eye state detection.