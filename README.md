# Adaptive Finite Difference-Based Multi-Grade Deep Learning (FD-MGDL)
Codes associated with the manuscript titled "The Adaptive Solution of High-Frequency Helmholtz Equations via Multi-Grade Deep Learning" authored by Peiyao Zhao, Rui Wang, Tingting Wu ang Yuesheng Xu. In this work, we compare our proposed methods, FD-MGDL and FD-SGDL, with several baseline approaches: Mscale [1], FBPINN [2], SIREN [3], PINN [4], and Pre-PINN [5]. The source code for these comparative methods is publicly available at the following repositories:
- Mscale: https://github.com/xuzhiqin1990/mscalednn
- FBPINN: https://github.com/benmoseley/FBPINNs
- SIREN: https://github.com/vsitzmann/siren
- PINN: https://github.com/maziarraissi/PINNs

# Abstract
The Helmholtz equation is fundamental to wave modeling in acoustics, electromagnetics, and seismic imaging, yet high-frequency regimes remain challenging due to the ``pollution effect''. We propose FD-MGDL, an adaptive framework integrating finite difference schemes with Multi-Grade Deep Learning to efficiently resolve high-frequency solutions. While traditional PINNs struggle with spectral bias and automatic differentiation overhead, FD-MGDL employs a progressive training strategy, incrementally adding hidden layers to refine the solution and maintain stability. Crucially, when using ReLU activation, our algorithm recasts the highly non-convex training problem into a sequence of convex subproblems. Numerical experiments in 2D and 3D with wavenumbers up to $\kappa=200$ show that FD-MGDL significantly outperforms single-grade and conventional neural solvers in accuracy and speed. Applied to an inhomogeneous concave velocity model, the framework accurately resolves wave focusing and caustics, surpassing the 5-point finite difference method in capturing sharp phase transitions and amplitude spikes. These results establish FD-MGDL as a robust, scalable solver for high-frequency wave equations in complex domains.

**References**\
[1] Liu, Z., Cai, W., & Xu, Z.-Q. J. (2020). Multi-Scale Deep Neural Network (MscaleDNN) for Solving Poisson-Boltzmann Equation in Complex Domains. Communications in Computational Physics, 28(5), 1970–2001.\
[2] Moseley, B., Markham, A., & Nissen-Meyer, T. (2023). Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving differential equations. Advances in Computational Mathematics, 49(4), 62.\
[3] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., & Wetzstein, G. (2020). Implicit neural representations with periodic activation functions. Advances in Neural Information Processing Systems, 33, 7462–7473.\
[4] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686–707.\
[5] De Ryck, T., Bonnet, F., Mishra, S., & de Bézenac, E. (2024). An operator preconditioning perspective on training in physics-informed machine learning. Proc. Int. Conf. Learn. Represent., 54886–54914.
