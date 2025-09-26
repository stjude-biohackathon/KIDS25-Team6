# KIDS25-Team6 MolSnap

# ML Modelling
The implementation of the ML model to predict SMILES strings from single images is located in the ML_model subfolder.

## Model checkpoint 
Download the model checkpoint from https://zenodo.org/records/13304899/files/molnextr_best.pth?download=1

## Model fine-tuning
A training run can be launched by the following command.
Adapt the ```--load_path``` to point to a model checkpoint from which to load the weights, the ```--train_file``` to point to the csv to fine tune on, the ```--valid_file``` to point to the csv to fine validate on and  the ```--save_path``` option to define the output folder for the fine-tuned model.

```bash
 torchrun --nproc_per_node=1 --nnodes=1 --node_rank 0 --master_addr localhost --master_port 63868 ML_model/train.py \
 --data_path Training-Data/test_mini \
 --train_file test_mini_train.csv  \
 --coords_file aux_file \
 --valid_file test_mini_valid.csv \
 --vocab_file ML_model/MolNexTR/vocab/vocab_chars.json \
 --formats chartok_coords,edges \
 --dynamic_indigo --augment --mol_augment \
 --include_condensed \
 --coord_bins 64 \
 --sep_xy \
 --input_size 384 \
 --encoder_lr 4e-4 \
 --decoder_lr 4e-4 \
 --save_path output_path \
 --load_path molnextr_best.pth \
 --save_mode all \
 --label_smoothing 0.1 \
 --epochs 40 \
 --batch_size 32 \
 --gradient_accumulation_steps 1 \
 --use_checkpoint \
 --warmup 0.02 \
 --print_freq 200 \
 --do_train \
 --do_valid \
 --fp16 \
 --backend gloo
 ``` 

## Model evaluation
Coming soon

## Model inference interface
To predict SMILES from a single image file (png and jpg supported), interface as following

```python
from ML_model import prediction

results = prediction.predict_from_image_file('1.png','checkpoints/molnextr_best.pth')

```