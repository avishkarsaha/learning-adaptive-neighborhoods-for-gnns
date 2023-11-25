## Learning Adaptive Neighborhoods for Graph Neural Networks

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This repository contains a PyTorch implementation of the ICCV 2023 paper "[Learning Adaptive Neighborhoods for Graph Neural Networks](https://openaccess.thecvf.com/content/ICCV2023/html/Saha_Learning_Adaptive_Neighborhoods_for_Graph_Neural_Networks_ICCV_2023_paper.html)".

This repository is based on https://github.com/chennnM/GCNII, we use their 
repository structure and their training scripts. We thank the authors for their
code.



### Dependencies
- CUDA 10.1
- python 3.6.9
- pytorch 1.9.1
- networkx 2.1
- scikit-learn
- torch_geometric 2.1.0

### Datasets

 We have maintained the datasets from the GCNII repository, with the
 `data` folder containing three benchmark datasets(Cora, Citeseer, Pubmed).
 

### Models
We have integrated our Differentiable Graph Generator (DGG) into the following
graph neural network models:
1. GCNII
2. GAT
3. SAGE
4. GCN

These models are available in the ```models.py``` file. Pretrained models are
coming soon!

### Usage
To run the training pipelines, use the following command:
```
python train_small_graphs.py --dataset cora --model GCN_DGG 
```
The arguments for the DGG can be found in the training scripts. 

### Citation
```
@InProceedings{Saha_2023_ICCV,
    author    = {Saha, Avishkar and Mendez, Oscar and Russell, Chris and Bowden, Richard},
    title     = {Learning Adaptive Neighborhoods for Graph Neural Networks},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {22541-22550}
}
```
