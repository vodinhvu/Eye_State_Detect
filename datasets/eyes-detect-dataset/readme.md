# Eye Detection Dataset

This folder contains the resources and instructions for managing the dataset used in the Eye Detection project.

## Dataset Installation

1. Visit the dataset download link:  
   [Eyes3.0 Dataset - Roboflow](https://universe.roboflow.com/valerio-ponzi/eyes3.0/dataset/2)

2. Download the dataset in a format compatible with YOLOv8 (e.g., YOLO).

3. Extract the dataset contents into this folder.

## Folder Structure

Once the dataset is downloaded and extracted, the folder structure should look like this:

```plaintext
dataset/
├── train/
├── val/
├── test/
├── data.yaml
```
- train/: Contains training images and annotations.
- val/: Contains validation images and annotations.
- test/: Contains test images and annotations.
- data.yaml: The configuration file specifying paths to the dataset subsets.

**Note**: Ensure the dataset path is correctly referenced in your project configuration files.

