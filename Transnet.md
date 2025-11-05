## The Target Problem

Here  $\Omega \sub R^d$ , which is a spatial-temporal bounded domain

<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250923194543274.png" alt="image-20250923194543274" style="zoom:50%;" />

## **Neural Feature Space**

$P_{NN}=span\{1,\sigma(\omega_1^Ty+b_1,\cdots,\omega_M^Ty+b_M)\}$

Reparameterization of $P_{NN}$

$\begin{cases} \omega_m=\gamma_m a_m \\ b_m=\gamma_m r_m \end{cases}$     $\begin{cases} a_m=\frac{\omega_m}{||\omega_m||_2} \\ r_m=\frac{b_m}{||\omega_m||_2} \\ \gamma_m=||\omega_m||_2 \end{cases}$

## Transnet solution (modified version in Multi-transnet)

$u^{NN}=\sum_{m=1}^M\alpha_m\sigma(\gamma(a_m^T(x-x_c)+r_m)+\alpha_0$

$a_m$ is uniformly distributed in the unit sphere in $R^d$ and $r_m$ is uniformly distributed in $[0,R]$   , where the circle $(x_c,R)$ slightly cover $\Omega$

define the following posterior error indicator function  $\eta(\gamma)=min_{\alpha} Loss_{TN}(\alpha)$

use golden search to determine $\gamma$ by          $min_{\gamma}\ \ \eta(\gamma)$  where golden search is 
<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250907165146545.png" alt="image-20250907165146545" style="zoom:50%;" />

Where <img src="/Users/syh/Library/Application Support/typora-user-images/image-20250907165034052.png" alt="image-20250907165034052" style="zoom:50%;" />

for the question <img src="/Users/syh/Library/Application Support/typora-user-images/image-20250907165109016.png" alt="image-20250907165109016" style="zoom:50%;" />

Finally we get $\alpha$ by turn the question into a least square problem $min_{\alpha} ||F\alpha-T||^2$

## Selecting procedure in paper

文章中步骤：

1. 生成$k$个GRF实现$G(y|\omega_k,\eta)$
2. 在单位球内选取$J$个均匀采样点$y_j$
3. 对每个候选的$\gamma$（在一维网格上搜索）：
   - 用已采样好的位置参数 $(a_m,r_m)$ 和该 $\gamma$ 构建出基函数 $\psi_m(y)=\sigma(\gamma (a_m^⊤y+r_m))$
   - 对第$k$个 GRF 实现的数据$\{y_j,g_j^k\}$用最小二乘法拟合系数 $\alpha$
   - 计算拟合的 MSE
4. 对 $k$个 GRF 实现的 MSE 取平均，选择平均 MSE 最小的 $\gamma$ 作为最优的 $\gamma_{opt}$。

可以把 GRF 看作**一组用于校准的“标准测试函数”**：

- 如果特征空间能很好地拟合这些具有不同空间变化模式的随机函数，那么它应对未知的 PDE 解也会有较好的表达能力。
- 相关长度 $\eta$ 相当于控制测试函数的“难度”：$\eta$ 小 → 高频变化多 → 需要更窄的过渡带（较大的 $\gamma$）才能捕捉细节。
- 这样调出的 $\gamma$ 不是针对某一个 PDE，而是针对**一类可能的空间变化模式**，从而保证迁移性。



## Content in Transnet

<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250926103728563.png" alt="image-20250926103728563" style="zoom:50%;" />

<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250926113324513.png" alt="image-20250926113324513" style="zoom:50%;" />

解释：对任意一个满足 $||y|| \leq 1−\tau||y|| \leq 1−\tau$ 的点 $y$，随机采样的超平面中，期望上有比例 $\tau$ 的超平面距离 $y$ 小于 $\tau$。

也就是说，在输入空间的单位球内部（离边界至少 $\tau$），点 $y$ 周围 $\tau$ 范围内的超平面密度期望值正好是 $\tau$

### Gaussian random field

Def: Let $G$ be a countable set. The family of random variables $\{X_n\}_{n\in G}$ is called a Gaussian Random Field (GRF), if for any finite subset $\{n_1,\cdots,n_k\} \sub G$, the random variables $\{X_{n_1},\cdots,X_{n_k}\}$ are jointly Guassian.



## Limitations

1. **维数灾难（Curse of Dimensionality）**
   - **位置**：在第10页的 **Remark 1** 中明确提到。
   - **内容**：尽管论文的理论证明适用于任意维度，但作者承认，要有效覆盖高维单位球所需的神经元数量可能仍然是“难以处理的”。这意味着TransNet方法主要适用于科学和工程中常见的**低维PDE**（例如，3D空间+1D时间，共4维）。
   - **解读**：这是所有基于全局基函数的谱方法共有的问题。当输入维度（空间+时间）很高时，需要指数级增长的神经元数量来保持分辨率，这使得方法在高维问题上计算成本高昂。
2. **仅限于浅层网络（Single-hidden-layer Network）**
   - **位置**：在第20页的 **第4节“结论与未来工作”** 中作为未来研究方向提出。
   - **内容**：论文的工作仅限于单隐藏层的神经网络。作者指出，虽然对于文中所测试的PDE来说表达能力已经足够，但对于更复杂的PDE（例如湍流模型），可能需要具有更高表达能力的**多层网络**。
   - **解读**：将TransNet的思想（构建可迁移特征空间）扩展到深度网络是一个非平凡的问题，也是未来的挑战。目前的方法无法利用深度网络可提供的更强大的特征提取能力。
3. **最小二乘系统的潜在数值问题（Potential Numerical Issues in Least Squares）**
   - **位置**：同样在第20页的 **第4节** 中作为未来研究方向提出。
   - **内容**：作者提到，由于神经元构成的基函数是**非正交**的，有可能出现神经元之间线性相关的情况。这会降低最小二乘矩阵的列秩，甚至导致系统是欠定的（方程数少于变量数）。
   - **解读**：这暗示了在实际应用中，当神经元数量`M`很大或采样点选择不当时，直接求解最小二乘可能不稳定。论文建议未来可能需要引入正则化技术（如岭回归）来稳定求解过程。
4. **离线计算成本（Offline Computational Cost）**
   - **位置**：在第13页的 **Remark 2** 中。
   - **内容**：虽然调参（寻找最优的`γ`）是离线完成的，但其计算成本可能很高。这个过程需要解决大量（`S × K`个）最小二乘问题。`S`是`γ`的搜索网格大小，`K`是GRF实现的数量。
   - **解读**：尽管这是一次性的前期投入，但对于非常大的`M`和精细的网格搜索，这个离线阶段的成本不容忽视。作者建议使用插值或更好的优化算法（如二分法）来减轻这一负担。
5. **相关长度η的选择（Choice of Correlation Length η）**
   - **位置**：在第13页的 **Remark 3** 中。
   - **内容**：GRF的相关长度`η`是一个需要预先设定的超参数。论文提出了两种策略：一是利用关于PDE解的先验知识（例如，低雷诺数流动不会有高频振荡）；二是使用一个“过杀”的较小`η`值来确保特征空间有足够的表达能力。
   - **解读**：这本身就是一个不确定性。如果对目标PDE的解缺乏先验知识，`η`的选择可能带有试探性，从而影响所构建特征空间的通用性。



总而言之，这篇文章非常诚实地指出了TransNet方法的局限性，主要集中在：

- **适用维度**： 主要针对**低维**PDE问题。
- **网络结构**： 目前仅限于**浅层**网络。
- **数值稳定性**： 最小二乘求解可能因基函数的非正交性而出现**病态问题**。
- **前期成本**： 确定最优形状参数`γ`的**离线调参过程**计算量较大。
- **超参数选择**： GRF的**相关长度η**需要合理选择。

这些局限性也为作者及其后续研究者指明了清晰的未来工作方向，如扩展至深度学习、理论分析收敛性、改进最小二乘求解稳定性等。



目前可尝试的任务：

1. 将单隐藏层的transnet加一层神经网络，观察是否有更强大的能力
2. 神经元之间可能会出现线性相关的情况，需要解决





神经网络结构：

以transnet先做一个基准解，然后套上几层神经网络，最后再用最小二乘



可以解非线性算子，需要在问题中使用迭代算法，$e.g.$ Pichard Iteration
