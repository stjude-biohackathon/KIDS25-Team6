# KIDS25-Team6 MolSnap

# How to run MolSnap locally

## Download Model Checkpoint
Download the model checkpoint from https://zenodo.org/records/13304899/files/molnextr_best.pth?download=1

## Fine-tune on specific molecule types
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
## Setting up the conda environments
We will need two conda environments, `decimer` and `molsnap`

### Creating `decimer`
```bash
# create and activate env
conda create -n decimer python=3.10
conda activate decimer

# OPTIONAL: if pip is not installed in your conda environment
conda install pip
python -m pip install -U pip

# clone the DECIMER Image Segmentation repo
git clone https://github.com/Kohulan/DECIMER-Image-Segmentation.git

# assuming pip is installed,
cd DECIMER-Image-Segmentation
pip install .
pip install decimer-segmentation

# go to decimer-api/ directory
cd ../decimer-api/
pip install -r requirements.txt
```

### Creating `molsnap`
```bash
# create and activate env
conda create -n molsnap python=3.10
conda activate molsnap

# OPTIONAL: if pip is not installed in your conda environment
conda install pip
python -m pip install -U pip

# assuming you're in the root: KIDS25-Team6/
pip install -r requirements.txt

# go to molsnap-api/ directory
cd molsnap-api/
pip install -r requirements.txt
```

## Running MolSnap
If you don't have Node.js installed, install nvm. Follow installation guidelines here: http://github.com/nvm-sh/nvm
```bash
# in one terminal tab
conda activate decimer # ensure you're using decimer env
cd KIDS25-Team6/decimer-api
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# open another terminal tab, and 
conda activate molsnap # ensure you're using molsnap env
cd KIDS25-Team6/
uvicorn "molsnap-api.main:app" --host 0.0.0.0 --port 8000 --reload

# open a third terminal tab, and
cd KIDS25-Team6/molsnap
npm install
npm run dev
```
Application is now running on http://localhost:5173/