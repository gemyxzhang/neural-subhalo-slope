# Inferring subhalo effective density slopes from strong lensing observations with neural likelihood-ratio estimation
[![arXiv](https://img.shields.io/badge/arXiv-2208.13796%20-green.svg)](https://arxiv.org/abs/2208.13796)

## Software dependencies
The code uses standard `numpy` and `scipy` packages. Part of the forward model requires installation of the [sbi](https://github.com/mackelab/sbi) package. We use [paltas](https://github.com/swagnercarena/paltas) and [lenstronomy](https://github.com/lenstronomy/lenstronomy) for data generation. These can be installed as follows: 

```
pip install paltas lenstronomy 
```
We ran our analysis with `Python 3.7.7` and `Pytorch 1.11.0+cu102`. 


## Code 
To generate mock lensing images, use the following scripts (which have dependency on [utils.py](utils.py)): 
- [make_images_eplsh.py](make_images_eplsh.py) makes lensing images with EPL subhalos
- [make_images_nfwsh.py](make_images_nfwsh.py) makes lensing images with NFW subhalos

Note: if these scripts are run with slurm job arrays, then it is necessary to manually combine the gamma parameter files produced by the job arrays.

The likelihood-ratio estimator model class is in [resnet.py](resnet.py). To train the model on generated images, run [train.py](train.py) with the specified parameters (which has dependecy on [data_utils.py](data_utils.py)). 

[figures.ipynb](figures.ipynb) contains code that produces the plots in [2208.13796](https://arxiv.org/abs/2208.13796). 
