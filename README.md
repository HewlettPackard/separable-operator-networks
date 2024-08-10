# Separable Operator Networks (SepONet)

This is the official repository for separable operator networks (SepONet) originally introduced in [this preprint](https://arxiv.org/abs/2407.11253) [1]. 

## Installation

Please clone the project and install via pip:
```bash
git clone https://github.com/HewlettPackard/separable-operator-networks.git  
cd separable-operator-networks  
pip install -e .
```
Note that this repository uses JAX as a dependency. It is recommended to [install with CUDA compatibility](https://jax.readthedocs.io/en/latest/installation.html) prior to installing this library.

## Description

Operator learning has become a powerful tool in machine learning for modeling complex physical systems governed by partial differential equations (PDEs). Although Deep Operator Networks (DeepONet) show promise, they require extensive data acquisition. Physics-informed DeepONets (PI-DeepONet) mitigate data scarcity but suffer from inefficient training processes. We introduce Separable Operator Networks (SepONet), a novel framework that significantly enhances the efficiency of physics-informed operator learning. SepONet uses independent trunk networks to learn basis functions separately for different coordinate axes, enabling faster and more memory-efficient training via forward-mode automatic differentiation. Our preprint provides theoretical guarantees for SepONet using the universal approximation theorem and validate its performance through comprehensive benchmarking against PI-DeepONet. Our results demonstrate SepONet's superior performance across various PDEs. For the 1D time-dependent advection equation, SepONet achieves up to 112x faster training and 82x reduction in GPU memory usage compared to PI-DeepONet, while maintaining comparable accuracy. For more challenging problems, SepONet's advantages become more pronounced. In the case of the 2D time-dependent nonlinear diffusion equation, SepONet efficiently handles the complexity, achieving a 6.44% mean relative $\ell_{2}$ error on 100 unseen initial conditions, while PI-DeepONet fails due to memory constraints. This work paves the way for extreme-scale learning of continuous mappings between infinite-dimensional function spaces.

The SepONet architecture is depicted for $d=2$ dimensional coordinate grid below. Predictions of parametric PDEs are obtained efficiently by feeding points along coordinate axes through independent trunk networks, then performing cross product and sum-reduction with outputs of the branch network. Partial derivatives of outputs are obtained efficiently by forward automatic differentiation. The architecture is inspired by the method of separation of variables and recent exploration of [separable physics-informed neural networks](https://arxiv.org/abs/2211.08761) [2] for single instance PDE solutions.

![SepONet architecture for 2 dimensional coordinate grid](docs/assets/SepONet_Architecture.png?raw=true)

## Code Overview

Source code can be imported via  
```python
import separable_operator_networks  
```
SepONet and DeepONet models are implemented in the `models` submodule. PDE instances and helper functions can be imported from the `pde` submodule.

Test data can be generated using the Python scripts in scripts/generate_test_data. Test cases can be ran using the scripts in scripts/main_scripts and scripts/scale_tests.

## Authors

Sean Hooten (sean dot hooten at hpe dot com)  
Xinling Yu (xyu644 at ucsb dot edu)

## License

MIT (see LICENSE.md)

## References

[1] X. Yu, S. Hooten, Z. Liu, Y. Zhao, M. Fiorentino, T. Van Vaerenbergh, and Z. Zhang. Separable Operator Networks. arXiv preprint arXiv:2407.11253 (2024).  
[2] J. Cho, S. Nam, H. Yang, S.-B. Yun, Y. Hong, E. Park. Separable PINN: Mitigating the Curse of Dimensionality in Physics-Informed Neural Networks. arXiv preprint arXiv: 2211.08761 (2023).