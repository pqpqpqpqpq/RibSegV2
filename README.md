# Entropy-Guide Partial Annotation for Cross-Domain Rib Segmentation

## 实验环境
1. 创建虚拟环境
```bash
conda create -n ribseg python=3.7
conda activate ribseg
```
2. cd到项目路径，安装依赖库
```bash
pip install -r requirements.txt
```

## 预训练
执行data.data中的save_cube_nii()处理无标签数据
```bash
python pretrain.py
```

## 微调
执行data.data中的save_cube_label_nii()处理标签数据
```bash
python fintune.py
```

## 跨域训练
与微调类似，先处理数据
```bash
python domain.py
```