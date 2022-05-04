# Wasserstain Distance approximation with GAN network

Does the Wasserstain GAN trick to approximate the Wasserstain distance really works?  
You can check our implementation of wasserstain distance approximation by different variaties of WGAN's in pytorch/jax to help yourself with the answer!  


#### Structure
```
.
├── notebooks                   
│   ├── wasser.ipynb          # Pytorch implementation of Weight Clipping, GP, (c)-GAN and (c, eps)-GAN     
│   └── wasser_exp.ipynb      # Experiments with densities and width of hidden layer  
├── src                    
│   ├── torch.py              # Silkhorn and WeightClipping in Torch
│   └── utils.py              # Utils functions 
├── poetry.lock
├── pyproject.toml
└── README.md

```

#### Installation
To install all libraries needed you will need:
 - `poetry` ([source](https://python-poetry.org/))
 - `pyenv`  ([source](https://github.com/pyenv/pyenv))
 
Once both are install on your system follow thoses instructions:

```
cd OT_projet
poetry install
poetry run jupyterlab
```

#### Bibliography

This repo is heavly inspired by this [paper](https://arxiv.org/abs/1910.03875).

For the schemes: 
 - [Weight Clipping](https://arxiv.org/pdf/1701.07875.pdf)
 - [GP](https://arxiv.org/pdf/1704.00028.pdf)
 - [(c)-GAN](https://arxiv.org/pdf/1902.03642.pdf)
 - [(c, eps)-GAN](https://arxiv.org/pdf/1902.03642.pdf)  

Project by  
[Guillaume Peltier](https://github.com/g-peltier)  
[Vladimir Kondratyev](https://github.com/VldKnd)