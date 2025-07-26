# Fracture_Bone_Prediction

This repository contains a computer vision project implemented in Python using Jupyter Notebook and Streamlit for deployment. The goal of this project is to analyze and process images from fracture bone datasets to achieve a specific object detection task 


You can download dataset here [Dataset](https://drive.google.com/drive/folders/1h5lIBfUuc8mnh2PIxwXOUJloqk4ciSMj?usp=sharing)
Weight model: [Weight](https://drive.google.com/drive/folders/1bHws5HjavQFbnSz0EPteM1r5_Of1ngzF?usp=sharing)

```
├── Streamlit/ # Streamlit app files for the web interface 
├── ComputerVision.ipynb # Jupyter Notebook with the main code # This file
```

## 📌 Requirements

Make sure to install the required libraries before running the project:

```bash
pip install -r requirements.txt
```

## Optional: Create Virutal Environment

> In the directory containing the project, run the following command to create the virtual environment `venv`:

```bash
python -m venv myenv
```

> After creating the virtual environment, activate it with the command

```bash
myenv\Scripts\activate
```

> You can also disable venv if needed with the command

```bash
deactivate
```

### :rocket: How to run

1. Clone the repository:
``` bash
git clone https://github.com/VietHoang0805/Fracture_Bone_Prediction.git
cd Fracture_Bone_Prediction
```

2. Run the notebook:

+ Open [Model RESNET18](resnet-18-reduceoverfit.ipynb) in Jupyter Notebook or JupyterLab.

+ Open [Model VIT](vit-fracture-prediction-newversion.ipynb) in Jupyter Notebook or JupyterLab.

+ Open [Model DENSENET](densenet-bone-fracture.ipynb) in Jupyter Notebook or JupyterLab.

+ Open [Model MOBILENET](mobilenet-bone-fracture.ipynb) in Jupyter Notebook or JupyterLab.

You can see input image

![alt text](Images/Untitled.png "Title")

3. Launch the Streamlit app:
``` bash
streamlit run Streamlit/app.py
```
After run [app.py](Streamlit/app.py "Streamlit") you can see UI below

Image UI here: [StreamlitUI](/Images/Streamlit.png) 

![alt text](/Images/Streamlit.png "Streamlit")

You can use [Image Test](Images/Test) or other to test

4. Result model:


| Model     | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| ResNet18  | 94.07%   | 94.13%    | 93.98% | 94.04%   |
| MobileNet | 90.51%   | 90.62%    | 90.36% | 90.45%   |
| DenseNet  | 98.22%   | 98.20%    | 98.23% | 98.22%   |
| ViT       | 96.25%   | 96.21%    | 96.27% | 96.23%   |
