# Entropy-Guide Partial Annotation for Cross-Domain Rib Segmentation

## Experimental environment
1. Create Virtual Environment
```bash
conda create -n ribseg python=3.7
conda activate ribseg
```
2. Navigate to the project directory and install required libraries
```bash
pip install -r requirements.txt
```

## Pre-training
Data Preprocessing: Execute save_cube_nii() from data.data to process unlabeled data.
```bash
python pretrain.py
```

## Fine-tuning
Data Preprocessing: Execute save_cube_label_nii() from data.data to process labeled data.
```bash
python finetune.py
```

## Cross-Domain Training
Training: Follow the same protocol as fine-tuning, but with target-domain data.
```bash
python domain.py
```