# KIDS25-Team6 MolSnap

# ML Modelling
The implementation of the ML model to predict SMILES strings from single images is located in the ML_model subfolder.

## Model checkpoint 
Download the model checkpoint from https://zenodo.org/records/13304899/files/molnextr_best.pth?download=1

## Model fine-tuning
```bash
torchrun  --nproc_per_node=1 --nnodes=1 --node_rank 0 --master_addr localhost --master_port 10042 ML_model/train.py --train_file test_mini.csv --data_path Training-Data/test_mini --valid_file test_mini.csv
``` 

## Model evaluation
Coming soon

## Model inference interface
To predict SMILES from a single image file (png and jpg supported), interface as following

```python
from ML_model import prediction

results = prediction.predict_from_image_file('1.png','checkpoints/molnextr_best.pth')

```