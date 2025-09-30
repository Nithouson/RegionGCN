# RegionGCN

This repository contains the implementation and data for the paper:
>**RegionGCN: Spatial-Heterogeneity-Aware Graph Convolutional Networks**
>
> Hao Guo, Han Wang, Di Zhu, Lun Wu, A. Stewart Fotheringham, Yu Liu
>  
>**Abstract:** Modeling spatial heterogeneity in the data generation process is essential for understanding and predicting geographical phenomena.
> Despite their prevalence in geospatial tasks, neural network models usually assume spatial stationarity, which could limit their performance in the presence of spatial process heterogeneity.
> By allowing model parameters to vary over space, several approaches have been proposed to incorporate spatial heterogeneity into neural networks.
> Current geographical weighting approaches, however, are ineffective on graph neural networks, yielding no significant improvement in prediction accuracy.
> We assume the crux lies in the overfitting risk brought by a large number of local parameters.
> Accordingly, we propose to model spatial process heterogeneity at the regional level rather than at the individual level, which largely reduces the number of spatially varying parameters.
> We further develop a heuristic optimization procedure to learn the region partition adaptively in the process of model training.
> Our proposed spatial-heterogeneity-aware graph convolutional network, named RegionGCN, is applied to the modeling of county-level vote share in the 2016 U.S. presidential election based on socioeconomic attributes.
> Results show that RegionGCN achieves significant improvement over the basic and geographically weighted GCNs.
> We also offer an exploratory analysis tool for the spatial variation of nonlinear relationships through ensemble learning of regional partitions from RegionGCN.
> Our work contributes to the practice of geospatial artificial intelligence in tackling spatial heterogeneity.

[\[Full-text at the publisher\]](https://doi.org/10.1080/24694452.2025.2558661) [\[arXiv\]](https://arxiv.org/abs/2501.17599)

## File description


## Workflow


## Citation
If you use this code in your research, please cite:

```bibtex
@article{Guo29092025,
author = {Hao Guo and Han Wang and Di Zhu and Lun Wu and A. Stewart Fotheringham and Yu Liu},
title = {RegionGCN: Spatial-Heterogeneity-Aware Graph Convolutional Networks},
journal = {Annals of the American Association of Geographers},
volume = {0},
number = {0},
pages = {1--17},
year = {2025},
publisher = {Taylor \& Francis},
doi = {10.1080/24694452.2025.2558661}
}
```

## Contact
If you have any questions, feel free to contact us through email sinesloop@pku.edu.cn.
