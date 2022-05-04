# Wasserstain Distance approximation
Does the Wasserstain GAN trick to approximate the Wasserstain distance really works?
You can check our implementation of wasserstain distance approximation by different variaties of WGAN's in pytorch/jax to help yourself with the answer!

_Structure_  
wasser_estimation.ipynp - JAX implementation of WGAN with [Weight Clipping](https://arxiv.org/pdf/1701.07875.pdf)/[GP](https://arxiv.org/pdf/1704.00028.pdf)/[(c)-GAN](https://arxiv.org/pdf/1902.03642.pdf)/[(c, eps)-GAN](https://arxiv.org/pdf/1902.03642.pdf)  
wasser_estimation_torch.ipynb  - Pytorch implementation of WGAN with [Weight Clipping](https://arxiv.org/pdf/1701.07875.pdf)/[GP](https://arxiv.org/pdf/1704.00028.pdf)/[(c)-GAN](https://arxiv.org/pdf/1902.03642.pdf)/[(c, eps)-GAN](https://arxiv.org/pdf/1902.03642.pdf)  
wasser_estimation_torch_experiments.ipynb  - Experiments with densities and width of hidden layer  
src/  
&nbsp;&nbsp;&nbsp;&nbsp; /\_\_init\_\_.py - empty  
&nbsp;&nbsp;&nbsp;&nbsp; /torch.py - Pytorch implementation of Sinkhorn algorithm and Weight Clipping


Project by  
[Guillaume Peltier](https://github.com/g-peltier)  
[Vladimir Kondratyev](https://github.com/VldKnd)