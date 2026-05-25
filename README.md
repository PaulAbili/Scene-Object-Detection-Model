# Scene and Object Detection Model
[Site Link](https://scene-object-detection-model.streamlit.app/)

### Overview 
This project was created for UMBC's CMSC 475 class: Neural Networks.<br>

Starting with a pretrained [YOLO11N](https://docs.ultralytics.com/models/yolo11), we finetuned the model to increase the number of objects detectable from 80 to 349. Additionally we trained a scene classifer using the weights from the best version of our object detection model and a created [custom classification head](https://github.com/SmilingSupernova/Scene-Object-Detection-Model/blob/main/Model/scenes_classifier.py). Our scene classifer can detect 365 scenes. <br>

Each model was trained for 15 epochs.<br>

You can find more infomation on our [final report](https://github.com/SmilingSupernova/Scene-Object-Detection-Model/blob/main/Final%20Report.pdf)
### Installation and Setup
We utilized the following imports:
```python
import streamlit as st
import torch
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
```

These pip installs are necessary to run locally
```bash
pip install streamlit
pip install torch
pip install pillow
pip install ultralytics
pip install torchvision
``` 
or one could simply run

``` bash
pip install -r requirements.txt
```

to run its simply
``` python 
streamlit run streamlit_app.py
```
### Directories

#### Models:
Contains files used to train and finetune model.

#### UI:
Contains streamlit UI file, both model pt files and test_images.

### Limitations
1) Unevenly biased dataset
    - The # of training per class are very uneven. The person class dominates both the pre-trained data and the fine-tuned data. This leads to items with very few training examples, such as a toaster, often often being misclassifed if classified at all. 
2) Training Epochs
    - Each model was only trained 15 epochs. Throughout the course of this project, UMBC's Chip was experiencing a lot of issues causing many training epochs to be dropped midway. Ultimately, we were completely unable to fine-tune our object detection model on chip and used colab (this was not free). This lead to in general low confidence levels for classifications even with a clear image, and an increase in misclassifications and even duplicate classifications (classifying multiple parts of the object as seperate objects)
3) Ambigious Background
    - A lot of Images have an ambigious, unidentifiable or even completely lack a background. These cases lead to misclassifications by the scene classifier. These cases may even affect the object detection model.   

### Authors
- **Paul Abili** - ([@PaulAbili](https://github.com/PaulAbili)) 
    - Email: pabili1@umbc.edu
- **Alex Gallagher** - ([@AG446](https://github.com/AG446)) 
    - Email: a446@umbc.edu
- **Devon Kaelberer** - ([@Pie315](https://github.com/Pie315)) 
    - Email: devonk2@umbc.edu
- **Mark Quimba** - ([@mclquimba](https://github.com/mclquimba)) 
    - Email: xj44416@umbc.edu
- **Khushika Shah** - ([@Khushika1805](https://github.com/Khushika1805)) 
    - Email: kshah7@umbc.edu

### References
[Objects365](https://www.objects365.org/overview.html)
[Places365](https://docs.pytorch.org/vision/0.17/generated/torchvision.datasets.Places365.html)
[YOLO11N](https://docs.ultralytics.com/models/yolo11)
[Report](https://github.com/SmilingSupernova/Scene-Object-Detection-Model/blob/main/Final%20Report.pdf)
