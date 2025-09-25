# KIDS25-Team6 MolSnap

# ML Modelling
The implementation of the ML model to predict SMILES strings from single images is located in the ML_model subfolder.

## Model checkpoint 
Download the model checkpoint from https://zenodo.org/records/13304899/files/molnextr_best.pth?download=1

## Model fine-tuning
Coming soon

## Model evaluation
Coming soon

## Model inference interface
To predict SMILES from a single image file (png and jpg supported), interface as following

```python
from ML_model import prediction

results = prediction.predict_from_image_file('1.png','checkpoints/molnextr_best.pth')

```