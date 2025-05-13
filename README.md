# Fracture_Bone_Prediction

This repository contains a computer vision project implemented in Python using Jupyter Notebook and Streamlit for deployment. The goal of this project is to analyze and process images from fracture bone datasets to achieve a specific object detection task 


You can download dataset here [Dataset](https://drive.google.com/drive/folders/1h5lIBfUuc8mnh2PIxwXOUJloqk4ciSMj?usp=sharing)

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

Open [ComputerVision.ipynb](ComputerVision.ipynb) in Jupyter Notebook or JupyterLab.

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


| Model       | Accuracy (Train) | Accuracy (Test)   |
| ----------- | ---------------- |-----------------  | 
| VIT         |   89.14%         |         88.53%    | 
| RESNET-18   |   94.69%         |         96.25%    |