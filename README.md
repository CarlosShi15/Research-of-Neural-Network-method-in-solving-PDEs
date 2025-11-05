# 🧠 TransNet Research Project

> **Neural-network–based PDE solver with analytic basis functions and ε-rank analysis**

This repository contains the implementation and extended studies of the **TransNet framework** for solving partial differential equations (PDEs).  
The project explores TransNet’s numerical behavior, scalability, and extensions — including a **Hybrid-TransNet architecture** and **iterative schemes for nonlinear operators**.

---

## 📘 Overview

**TransNet** represents the solution of a PDE as a linear combination of *analytic basis functions*:

\[
u(x) = \sum_{m=1}^M \alpha_m \, \psi_m(x; r_m, \gamma),
\]

where  
- \( \psi_m \) are differentiable basis functions (e.g., Gaussian, tanh, sinusoidal),  
- \( r_m \) are centers or projection directions,  
- \( \gamma \) controls the scale,  
- \( \alpha_m \) are optimized coefficients.  

The parameters are obtained by minimizing the **least-squares residual functional**

\[
L(\alpha, \gamma)
   = \|\mathcal{N}[u_{\alpha,\gamma}]\|^2_{\Omega}
   + \lambda \|u_{\alpha,\gamma} - g\|^2_{\Gamma}.
\]

This approach is **mesh-free**, interpretable, and analytically differentiable — offering a stable and efficient alternative to traditional finite-difference or finite-element methods.

---

## 🧩 Research Components

### 1. Dimensional Study

📂 `mse.ipynb` & 🖼 `Result.png` contain experiments analyzing **TransNet performance across dimensions**.

- The **mean-squared-error (MSE)** decay was tracked for 1-D and 2-D PDEs.  
- As dimensionality increased, convergence slowed and the residual plateaued higher — reflecting the *curse of dimensionality* and conditioning issues in the least-squares system.  
- Despite this, TransNet maintained stable convergence and accurate solutions even in 2-D settings.

> These experiments validated the scalability and robustness of TransNet beyond the 1-D benchmark problems.

---

### 2. Reproduction of the Original TransNet Method

The baseline TransNet implementation was **faithfully reproduced** from the original paper.

- **γ optimization** via **golden-section search**.  
- **α computation** via **linear least-squares**.  
- Benchmarks: Poisson and wave equations.  
- Verified convergence and MSE consistent with the reference results.  
- **ε-rank tracking** confirmed the *staircase phenomenon*:  
  loss decreases stepwise as the effective rank of the Gram matrix increases.

> ✅ Reproduction validates the theoretical connection between loss decay and ε-rank growth.

---

### 3. Hybrid-TransNet: Improved Expressive Power

A new **Hybrid-TransNet** architecture was proposed to combine analytical structure with data-driven adaptability.

\[
u(x) = u_{\text{TransNet}}(x) + \mathcal{N}_{\text{MLP}}(x; \theta),
\]

where  
- `TransNet` part provides smooth analytical approximation,  
- `MLP` part captures nonlinear or high-frequency residuals.

**Key features**
- Skip connections (“Hybrid-TransNet-Skip”) improve gradient flow.  
- Outperforms baseline TransNet in both interior and boundary MSE.  
- Faster and larger ε-rank growth → richer feature representation.

> ⚡ Hybrid-TransNet bridges analytic theory and deep learning flexibility.

---

### 4. Iterative Treatment for Nonlinear Operators

For nonlinear PDEs (e.g., with terms like \( u \nabla u \)),  
a **Picard-type iterative scheme** was incorporated:

1. Initialize \( u^{(0)}(x) = 0 \).  
2. Linearize the nonlinear operator around \( u^{(k)} \).  
3. Solve for \( u^{(k+1)} \) using the TransNet least-squares formulation.  
4. Repeat until the residual converges.

**Advantages**
- Avoids instability of nonlinear residual minimization.  
- Preserves analytical differentiability.  
- Works efficiently for nonlinear Burgers-type equations.  

> 🌀 This extension allows TransNet to handle nonlinear PDEs while retaining its semi-analytic structure.

---

## ⚙️ Repository Structure

