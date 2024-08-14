# Separable Operator Networks (SepONet)
[![Static Badge](https://img.shields.io/badge/arXiv-2407.11253-blue?link=https%3A%2F%2Farxiv.org%2Fabs%2F2407.11253)](https://arxiv.org/abs/2407.11253)
[![Static Badge](https://img.shields.io/badge/pypi-v0.0.1-green?link=https%3A%2F%2Fpypi.org%2Fproject%2Fseparable-operator-networks%2F)](https://pypi.org/project/separable-operator-networks)

This is the official repository for separable operator networks (SepONet) originally introduced in [this preprint](https://arxiv.org/abs/2407.11253) [1]. 

## Installation
This code uses JAX as a dependency. It is recommended to [install with GPU/TPU compatibility](https://jax.readthedocs.io/en/latest/installation.html) prior to installing this library. JAX CPU is provided as the default dependency.

Please install with pip:
```bash
pip install separable-operator-networks
```
Alternatively, you may specify the `[cuda12]` extra to install `jax[cuda12]` automatically:
```bash
pip install separable-operator-networks[cuda12]
```

## Description

Operator learning has become a powerful tool in machine learning for modeling complex physical systems governed by partial differential equations (PDEs). Although Deep Operator Networks (DeepONet) show promise, they require extensive data acquisition. Physics-informed DeepONets (PI-DeepONet) mitigate data scarcity but suffer from inefficient training processes. We introduce Separable Operator Networks (SepONet), a novel framework that significantly enhances the efficiency of physics-informed operator learning. SepONet uses independent trunk networks to learn basis functions separately for different coordinate axes, enabling faster and more memory-efficient training via forward-mode automatic differentiation. The SepONet architecture for a $d=2$ dimensional coordinate grid is depicted below. The architecture is inspired by the method of separation of variables and recent exploration of separable physics-informed neural networks [2] for single instance PDE solutions.

Our [preprint](https://arxiv.org/abs/2407.11253) provides a universal approximation theorem for SepONet proving that it generalizes to arbitrary operator learning problems. For a variety of 1D time-dependent PDEs, SepONet has similar accuracy scaling to PI-DeepONet, but with as much as 112x faster training time and 82x reduction in GPU memory usage. For 2D time-dependent PDEs, SepONet is capable of accurate predictions at scales where PI-DeepONet fails. The full test scaling results as a function of the number of collocation points and number of input functions is shown below. These results may be reproduced using our [scripts](https://github.com/HewlettPackard/separable-operator-networks/tree/main/scripts).

![SepONet architecture for 2 dimensional coordinate grid](docs/assets/SepONet_Architecture.png?raw=true)
![Comparing SepONet to PI-DeepONet when varying number of collocation points](docs/assets/figure1_varying_Nc.png?raw=true)
![Comparing SepONet to PI-DeepONet when varying number of input functions](docs/assets/figure2_varying_Nf.png?raw=true)

## Code Overview

A SepONet model can be imported using:
```python
import jax
import separable_operator_networks as sepop
d = ... # replace with problem dimension
branch_dim = ... # replace with input shape for branch network (MLP by default)
key = jax.random.key(0)

model = sepop.models.SepONet(d, branch_dim, key=key)
```
Other model classes such as `PINN`, `SPINN`, `DeepONet` are implemented in the `sepop.models` submodule. These models are implemented as subclasses of `eqx.Module` (see [equinox](https://github.com/patrick-kidger/equinox)), enabling `eqx.filter_vmap` and `eqx.filter_grad`, along with easily customizable training routines via [optax](https://github.com/google-deepmind/optax) (see `sepop.train.train_loop(...)` for a simple `optax` training loop). PDE instances, loss functions, and other helper functions can be imported from the corresponding examples in the `sepop.pde` submodule (such as `sepop.pde.advection`).

Test data can be generated using the Python scripts in `/scripts/generate_test_data`. Test cases can be ran using the scripts in `/scripts/main_scripts` and `/scripts/scale_tests`.

## Citation

```tex
@misc{yu2024separableoperatornetworks,
title={Separable Operator Networks}, 
author={Xinling Yu and Sean Hooten and Ziyue Liu and Yequan Zhao and Marco Fiorentino and Thomas Van Vaerenbergh and Zheng Zhang},
year={2024},
eprint={2407.11253},
archivePrefix={arXiv},
primaryClass={cs.LG},
url={https://arxiv.org/abs/2407.11253}, 
}
```

## Authors

Sean Hooten (sean dot hooten at hpe dot com)  
Xinling Yu (xyu644 at ucsb dot edu)

## License

MIT (see LICENSE.md)

## References

[1] X. Yu, S. Hooten, Z. Liu, Y. Zhao, M. Fiorentino, T. Van Vaerenbergh, and Z. Zhang. Separable Operator Networks. arXiv preprint arXiv:2407.11253 (2024).  
[2] J. Cho, S. Nam, H. Yang, S.-B. Yun, Y. Hong, E. Park. Separable PINN: Mitigating the Curse of Dimensionality in Physics-Informed Neural Networks. arXiv preprint arXiv: 2211.08761 (2023).