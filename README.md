# GAMES103
HWs for GAMES103: Physics-Based Animation.

## Lab 1 - Angry Bunny (Rigidbody)

### Method 1. Impulse Method, Leapfrog Integration

![Rigidbody](pics/Rigidbody.gif)

### Method 2. Shape Matching

![Rigidbody-Shape Matching](pics/Rigidbody-ShapeMatching.gif)

## Lab 2 - Cloth Simulation 

### Method 1. Mass-spring Implicit Integration
#### CPU Implementation

![Mass-spring-CPU](pics/Mass-spring-CPU.gif)

#### GPU Parallelized Implementation (with Compute Shader)

![Mass-spring-GPU](pics/Mass-spring-GPU.gif)

### Method 2. Position Based Dynamics

![FVM-CPU-bang](pics/PBD.gif)

## Lab 3 - Bouncy House (Elasticbody Explicit FVM)

### Method 1. FVM with Green Strain

Slowed down, with Laplacian Smoothing:

![FVM-CPU](pics/FVM-CPU.gif)

Fail with large deformation:

![FVM-CPU-bang](pics/FVM-CPU-bang.gif)

### Method 2. FVM with Principal Invariants of Deformation Gradient (using SVD)
#### CPU Implementation

Slowed down, without Laplacian Smoothing:

![FVM-CPU-SVD-noLaplacian](pics/FVM-CPU-SVD-noLaplacian.gif)

#### GPU Parallelized Implementation (with CUDA<small>®</small>)
##### StVK Model

Without laplacian smoothing:

![FVM-GPU-StVK-noLaplacian](pics/FVM-GPU-StVK-noLaplacian.gif)

With laplacian smoothing:

![FVM-GPU-StVK-normal](pics/FVM-GPU-StVK-normal.gif)

Becomes plastic with large deformation:

![FVM-GPU-StVK-bang-clipped](pics/FVM-GPU-StVK-bang-clipped.gif)

##### neo-Hookean Model

Without laplacian smoothing:

![FVM-GPU-neoHookean](pics/FVM-GPU-neoHookean-noLaplacian.gif)

With laplacian smoothing:

![FVM-GPU-neoHookean-noLaplacian](pics/FVM-GPU-neoHookean-clipped.gif)

### Extra: Implicit Solver

Implemented according to [[Xu et al. 2018]](https://doi.org/10.1145/2766917).

Several detail issues remain to be solved...

![FVM-GPU-Implicit](pics/FVM-GPU-Implicit.png)

## Lab 4 - Pool Ripples (Shallow Wave)

### Water Drop

![ShallowWave-WaterDrop](pics/ShallowWave-WaterDrop.gif)

### Two-way Coupling with Rigidbodies

![ShallowWave-2way-initial](pics/ShallowWave-2way-initial.gif)

![ShallowWave-2way-drag](pics/ShallowWave-2way-drag.gif)

![ShallowWave-2way-drag](pics/ShallowWave-2way-draganddrop.gif)
