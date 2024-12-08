
# Eye State Detection  

This project is designed to detect the state of a person's eyes (open or closed) using a deep learning model running on a **Jetson Nano**.  

## Features  
- **Real-time Eye State Detection**: Processes video input to classify eye states as open or closed.  
- **YOLOv8 for Eye Detection**: Locates eyes in the video frames.  
- **PyTorch-based Eye State Classifier**: Classifies detected eyes as open or closed.  
- **Optimized for Jetson Nano**: Designed for resource-constrained environments.  
- **Training Scripts Included**: Includes code to train the YOLOv8 detection model and the eye state classifier.  

## Requirements  
### Hardware  
- Jetson Nano Developer Kit  
- Camera (e.g., USB or Raspberry Pi camera module)  

### Software  
- JetPack SDK (version 4.6 or later)  
- Python 3.8+  
- Docker (optional but recommended)  

### Dependencies  
Install the necessary libraries via `requirements.txt`:  
```bash  
pip install -r requirements.txt  
```  

### Major Libraries  
- `torch`, `torchvision` (PyTorch and vision models)  
- `ultralytics` (YOLOv8)  
- `opencv-python`  

## Project Structure  
```plaintext  
├── config/  
│   ├── settings.yml              # Configuration file (e.g., dataset URLs, parameters)  
├── src/                          # Code for inference and running the application  
│   ├── main.py                   # Entry point of the application  
│   ├── detector.py               # YOLOv8 detection logic  
│   ├── eye_classifier.py         # Classifies the state of eyes  
│   ├── utils.py                  # Utility functions  
├── datasets/                     # Folder for datasets (downloaded automatically)  
├── models/                       # Folder to store trained models  
│   ├── yolov8/                   # Pretrained YOLOv8 model files  
│   ├── torchvision/              # Pretrained eye state classification model files  
├── training/                     # Folder for training scripts  
│   ├── yolov8/                   # Code for training YOLOv8 eye detection model  
│   │   ├── train.py              # Script to train YOLOv8 model  
│   │   ├── config.yaml           # YOLOv8 training configuration  
│   │   ├── dataset/              # Folder for YOLOv8 dataset preparation  
│   │       ├── annotations/      # Annotation files for training  
│   │       ├── images/           # Image files for training  
│   └── eye_classification/       # Code for training the eye state classifier  
│       ├── train_classifier.py   # Training script for eye state classification  
│       ├── model.py              # Torchvision-based model definition  
│       ├── dataset.py            # Dataset loader and preprocessing logic  
│       ├── config.yml            # Configuration for training the classifier  
├── requirements.txt              # Python dependencies  
└── README.md                     # Project documentation  
```  

## Setup  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/yourusername/eye-state-detection.git  
   cd eye-state-detection  
   ```  

2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. Prepare the datasets:  
   - For YOLOv8: Place images and annotations in the `training/yolov8/dataset/` folder.  
   - For the classifier: Organize datasets for open and closed eye states in the format expected by `train_classifier.py`.  

## Training  
### Train YOLOv8 Model  
1. Navigate to the YOLOv8 training folder:  
   ```bash  
   cd training/yolov8  
   ```  
2. Start training:  
   ```bash  
   python train.py --config config.yaml  
   ```  

### Train Eye State Classifier  
1. Navigate to the classifier training folder:  
   ```bash  
   cd training/eye_classification  
   ```  
2. Start training:  
   ```bash  
   python train_classifier.py --config config.yml  
   ```  

## Running Inference  
Run the application using the pretrained models:  
```bash  
python src/main.py  
```  

## Deployment with Docker  
1. Build the Docker image:  
   ```bash  
   docker build -t eye-state-detection .  
   ```  
2. Run the container:  
   ```bash  
   docker run --rm --runtime nvidia -v $(pwd):/app eye-state-detection  
   ```  

## Acknowledgments  
This project utilizes NVIDIA Jetson Nano, YOLOv8, and PyTorch to deliver lightweight and efficient AI solutions.  
