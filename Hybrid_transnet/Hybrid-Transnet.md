# Hybrid-Transnet-NN

## Two ways

### 方法一：

Transnet后直接接MLP，$i.e.$
$u_{hybrid}(x)=MLP(z(x);\theta)\qquad z(x)=(\psi_1(x),\cdots,\psi_M(x))$

这里Transnet的作用是将输入$x \in R^d$，映射到一个 特征空间$z(x)=(\psi_1(x),\cdots,\psi_M(x))$，因此在这里它更像是一个特征提取器，是用来得到Neural feature space $P_{NN}$的。

$u_{hybrid}(x)=MLP(z(x);\theta)$，其中 $\theta$ 是 MLP 的所有权重与偏置，也就是我们要训练的参数

### 方法二：

Skip 接法：
$u(x)=u_{base}(x)+MLP(z(x))$

方法希望做到的：由TransNet 提供一个高精度、解析性强的“**主解**”，MLP 只负责学习 TransNet 无法很好表示的“**残差部分**”

### 方法三：

$u(x)=u_{base}(x)+v_{\theta}(x)$ , where 是一个小的 MLP 网络，用来“修正”主解没有拟合好的部分

$\begin{cases} −\Delta v_{\theta}(x)=f(x)+\Delta u_{base}(x)=:r_{PDE}(x) \quad\quad&\\ v_{\theta}(x)=0  \ \ \ \ \ \ \ \ x\in \partial \Omega \end{cases}$

这里$v_{\theta}(x)$设置在边界上为0是因为我们知道$u_{base}$关于真解的误差极小，因此可以认为为0

由(1)和(2)可以看出，$v_{\theta}$ 只负责“把 $u_{\text{base}}$ 没解释完的那一点点误差补上

## Experiment

### NO.1    Poisson case (2D)

$\begin{cases} -\Delta u(x)=f(x)&\ \Omega \\ u=g & \partial \Omega \end{cases}$      Here set $u(x)=sin(2\pi x_1)sin(2\pi x_2)$,  thus $f(x)=8\pi^2u(x)$

首先不受下面两方法影响，我们首先得到$\gamma_{opt} = 1.47063$

#### 方法一：

$u(x)=MLP(\psi_1(x),\cdots,\psi_M(x))$

<img src="/Users/syh/Library/Application Support/typora-user-images/image-20251015184023975.png" alt="image-20251015184023975" style="zoom:50%;" />

很明显效果非常差

尝试进行一些原因分析和思考：

1.针对$\gamma_{opt}$，我们在优化时是根据$min_{\alpha}\ Loss_{TN}(\alpha)$来进行优化的，但是我们在后面接了MLP后就不再存在计算线性的$\alpha$

2.观察训练过程，边界上的损失总是偏大，这说明MLP对边界的处理很不好

3.基线是 凸的线性最小二乘，一步到位给出全局最优 $\alpha$；但是本方法把“线性读出”换成了深度非线性的 θ，**变成强非凸问题**，容易卡在次优。



#### 方法二：

$u(x)=u_{base}(x)+MLP(z(x))$ , where $u_{base}$就是用transnet解出来的初始解

Experiment主要是进行Transnet（可认为成Baseline）和改进后的Hybrid方法的对比

先保证低维问题的可解性，再研究是否能克服高维难题

得到最终结果如下：

### 训练结果汇总

| 阶段               | 训练轮数 | 总损失    | PDE损失   | 边界损失  |
| ------------------ | -------- | --------- | --------- | --------- |
| **Baseline**       | -        | -         | -         | -         |
| **Hybrid训练开始** | 1        | 7.360e-05 | 3.292e-05 | 2.034e-05 |
|                    | 200      | 7.357e-05 | 3.292e-05 | 2.032e-05 |
|                    | 400      | 7.349e-05 | 3.284e-05 | 2.032e-05 |
|                    | 600      | 7.347e-05 | 3.283e-05 | 2.032e-05 |
|                    | 800      | 7.356e-05 | 3.291e-05 | 2.032e-05 |
|                    | 1000     | 7.335e-05 | 3.271e-05 | 2.032e-05 |
|                    | 1200     | 7.334e-05 | 3.271e-05 | 2.032e-05 |
|                    | 1400     | 7.334e-05 | 3.270e-05 | 2.032e-05 |
|                    | 1600     | 7.344e-05 | 3.281e-05 | 2.032e-05 |
|                    | 1800     | 7.354e-05 | 3.290e-05 | 2.032e-05 |
| **训练完成**       | 2000     | 7.345e-05 | 3.281e-05 | 2.032e-05 |

### 最终结果对比

| 方法          | MSE              | 优化参数            |
| ------------- | ---------------- | ------------------- |
| Baseline      | **2.509265e-06** | -                   |
| Hybrid (skip) | 2.541069e-06     | gamma_opt = 1.47057 |



这里最终结果对比，对比的是solution的MSE

<img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Transnet/Hybrid_transnet/results/loss output.png" style="zoom:72%;" />

![](/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Transnet/Hybrid_transnet/results/comparison.png)

以上结果发现hybrid最后的误差比transnet大，原因分析如下：

=== Checking PDE/BD loss terms === 
Baseline -> total=7.360e-05, pde=3.292e-05, bd=2.034e-05 
Hybrid   -> total=7.348e-05, pde=3.284e-05, bd=2.032e-05 
Correction norm = 2.965e-03, Baseline norm = 2.451e+01, ratio = 1.209e-04 === Refining Hybrid model with LBFGS === Refined Hybrid (LBFGS) MSE = 2.542929e-06

出现 Hybrid 误差不比 Baseline 更小，是因为 **Baseline 已经足够强**（LS 解几乎是解析最优），Hybrid 没法进一步改善。

修正项确实在起作用，但量级极小，所以最终效果和 baseline 几乎一样。

### NO.2    wave equation

系统A$\begin{cases} \dfrac{\partial^2 u}{\partial t^2}=c\dfrac{\partial^2 u}{\partial x^2} & x\in [0,1]\ \ t\in [0,2] \\ u(x,0)=sin(4\pi x) \\ u(0,t)=u(1,t) \end{cases}$                    系统B$\begin{cases} \dfrac{\partial^2 u}{\partial t^2}=c\dfrac{\partial^2 u}{\partial x^2} & x\in [0,1]\ \ t\in [0,2] \\ u(x,0)=sin(4\pi x)\quad \quad u_t(x,0)=0 \\ u(0,t)=u(1,t) \end{cases}$

这里虽然我们有了时间这一维度，但是在transnet看来还是2D，他并不能区分x或是t，因为输入的时候就是x的一列值罢了

**主解（TransNet base）**

- 用随机特征构造 $u_{base}$，解线性最小二乘问题
  $$
  \min_\alpha\big(\|u_{tt}-c\,u_{xx}\|^2 + \|u(x,0)-\sin(4\pi x)\|^2 + \|u(0,t)-u(1,t)\|^2\big)
  $$

- 如果加上 $u_t(x,0)=0$，就多一个初速项。

**残差网络 $v_\theta$**（Hybrid）

- 修正方程：
  $$
  v_{tt} - c v_{xx} = -\big(u_{base,tt}-c u_{base,xx}\big),
  \quad
  v(x,0)=0, \quad v(0,t)=v(1,t).
  $$



#### 与No.1的差异：

**方向向量**：$a_m=(a_x^{(m)},a_t^{(m)})$   $\xi=(x,t)$

**算子块：**$L(\psi)=\partial_{tt}\psi-c^2\partial_{xx}\psi=\sigma''(\gamma s)(a_t^2-c^2a_x^2)$   在代码中对应Psi_tt_int和 Psi_xx_int，
然后就对应了F_int = Psi_tt_int - c**2 * Psi_xx_int ，右端 T_int = 0

**约束块：** $\eta(\gamma)=min_{\alpha}(\lambda _L||u_{\alpha,tt}−c^2u_{\alpha,xx}||^2+\lambda _{IC}||\cdot||^2+\lambda_{BC}||\cdot||^2)$



#### 系统A和系统B的差别：

系统 A 是半约束的 PDE 拟合器」，系统 B 是完整的 Cauchy 初值问题求解器。

而在数值上，这个“多出来的一行约束矩阵”（即 $ψ_t(x,0)$ 块）起到了**决定性作用** ——它改善了条件数、锁定了传播方向、让 TransNet 的波形与解析解在整个时空域上对齐。

