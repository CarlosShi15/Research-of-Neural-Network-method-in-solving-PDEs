# Reproduction

复现过程中的一些理解：

文章目的是understand how neurons extract low dimensional features from high dimensional data

文章核心是提出$\epsilon$-rank这一指标,这一指标计算的是最后一层隐藏层的基函数的线性独立性，从而分析训练中的一些现象。第四章主要是通过例子以及不同的方法来分析神经元线性独立性的变化以及这些方法的影响。感觉本质上是一篇综述文章，大多是在引用其他文章的一些方法进行试验



**Method to compute $\epsilon$-rank locate in page 12 of the paper**

```compute_epsilon_rank
def compute_epsilon_rank(phi_matrix, g_values, eps=1e-3):
D = phi_matrix.detach().cpu().numpy()  # shape: (N, 100)
x = g_values.detach().cpu().numpy()
x = np.mean(x, axis=1)  # shape: (N,) — 用 g(x) 的均值排序
idx = np.argsort(x)
D = D[idx]
x = x[idx]
m = len(x)
dx = (x[-1] - x[0]) / (m - 1 + 1e-8)
w = np.ones(m)
w[0] = w[-1] = 0.5
w *= dx
W = np.diag(w)
M = D.T @ W @ D
M = (M + M.T) / 2  # 确保对称
eigvals = np.linalg.eigvalsh(M)
return np.sum(eigvals > eps)
```

其中 phi_matrix 是输出矩阵 ，g_values 充当了一个中间表示或投影特征，这个中间层的输出来定义样本的顺序，后续是分析在这种样本的排列下，待研究参数如何变化

值得注意的是，在第四章，由于要进行不用方法的对比，因此后续代码g_value改为了x，也就是按照网格顺序，也就是固定的，控制变量



## Example 2.6 (Function fitting)

1. different layers and different number of neurons  (use trapezoidal formula to calculate $\epsilon$​-rank)

   <div style="display: flex; justify-content: space-between;">
     <img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/L=2&n=50.png" style="width: 48%;" />
     <img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/L=2&n=25.png" style="width: 48%;" />
   </div>

   <div style="display: flex; justify-content: space-between;">
     <img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/L=4&n=50.png" style="width: 48%;" />
     <img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/L=4&n=25.png" style="width: 48%;" />
   </div>
   

2. Across different layers (n=50 L=4)

   To **match the vibration tendency**, I replace the optimizer Adam by SGD
   And to overcome the problem that mass matrix is sometimes hard to compute because $M\neq M^T$, I write $M=\dfrac{M+M^T}{2}$
   <img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/Different layers when L=4 n=50.png" style="zoom:72%;" />

   

3. Different activation functions

   <div style="display: flex; justify-content: space-between;">
     <img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/relu.png" style="width: 48%;" />
     <img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/elu.png" style="width: 48%;" />
   </div>

   <div style="display: flex; justify-content: space-between;">
     <img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/Cosine.png" style="width: 48%;" />
     <img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/tanh.png" style="width: 48%;" />
   </div>







## Chapter 4 $\epsilon$-rank in Existing methods

**4.1. Deterministic Initialization**

Example 4.1
initialization影响初始神经元的差异性，文章中deterministic是手动设置保证了神经元之间的差异性，因此初始$\epsilon$-rank就接近最大值
而random的设置，举例我现在代码中是正态分布，因此这种选择导致神经元差异性较小，最终出现了下面图像中情况

由于$\epsilon$-rank由于其定义，是研究神经元输出($\phi$-matrix)在数值意义上的线性独立程度。因此很自然的想到如何去捕捉这种线性独立性。文章此处实验就通过调整initialization得到对比结果进一步佐证了这一点。

For linearity of neurons, straightforward method: expand the range of uniform distribution             but $\to$  
exploding gradients and the saturation of activation functions

l=2 n=30

<div style="display: flex; justify-content: space-between;">
  <img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/initialization comparison rank(l=2 n=30).png" style="width: 48%;" />
  <img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/initialization comparison loss(l=2 n=30).png" style="width: 48%;" />
</div>

l=4 n=50

<div style="display: flex; justify-content: space-between;">
  <img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/initialization comparison rank(l=4 n=50).png" style="width: 48%;" />
  <img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/initialization comparison loss(l=4 n=50).png" style="width: 48%;" />
</div>


Example 4.2 PINN （UbI and Xavier)

UDI方法：

神经元第一层输出形式：
$$
\sigma(w_j^T x + b_j) = \sigma(\gamma_j (a_j^T x + r_j))
$$
**$a_j$**：方向向量，满足 $\|a_j\|^2 = 1$，即在**单位球面上均匀采样**（二维问题就是单位圆）。

**$r_j$**：偏移项，**在有界正实数区间上均匀采样**，区间和输入域 $\Omega$ 有关。

**$\gamma_j \ge 0$**：缩放系数（有时取 1，也可以作为可调参数）。

这种方法的目的是让第一层的超平面在输入域 $\Omega$ 中**均匀分布**，以保证表达能力。

复现中用了两层隐藏层和50个神经元

<div style="display: flex; justify-content: space-between;">
  <img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/initialization(pinn)rank.png" style="width: 48%;" />
  <img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/initialization(pinn)loss.png" style="width: 48%;" />
</div>


**4.2 Partial of Unity**

本节在实验中的区别在于是否采用了partition of unity ($i.e.$ PoU) technique

也就是将一个大区域划分成一些不相交的子区域，然后在子区域上进行计算

### RFM （PoU + 随机特征）

$$
u_R(x)
=\sum_{i=1}^{m}\psi_i(x)\sum_{j=1}^{J_R}u_{ij}\,\phi_{ij}(x)
$$

域 $\Omega$ 被划分为 $m$ 个**不重叠子区域** $\{\Omega_i\}_{i=1}^m$，

$\psi_i(x)=\mathbf 1_{\Omega_i}(x)$，即 PoU 函数只是“indicator”；

在每个子域 $\Omega_i$ 中，使用 $J_R$ 个随机特征 $\phi_{ij}(x)$；

权重 $u_{ij}$ 通过 **最小二乘法** 求解；

这里 m=9，J_R=100 ⇒ 总随机特征数 = 900

------

### ELM （无PoU）

$$
u_E(x) = \sum_{j=1}^{J_E} u_j\,\phi_j(x), \qquad J_E=900
$$

整个 $\Omega$ 不划分子域；

900 个随机特征在整个域上生成；

同样用最小二乘法求解输出层



<img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/RFMELM function fitting.png" style="zoom:72%;" />



下面对error进行分析

<img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/RFMELM error.png" style="zoom:67%;" />

<img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/RFMELM error and rank.png" style="zoom:67%;" />

可以看出RFM的rank明显比ELM高，error则差不多

但是error和激活函数有关，如果激活函数从tanh改为cos，误差会变的非常大。且ELM远大于RFM



4.3 ResNet

这部分内容借鉴了Residual Network的内容 , torch.nn里有residue block

### 残差学习 (Residual Learning)

1. 不直接学习目标映射 $H(x)$，而是学习一个 **残差函数**：

$$
F(x) = H(x) - x \quad \Rightarrow \quad H(x) = F(x) + x
$$

2. 如果最优映射接近于恒等映射，学习残差函数更容易。

3. **Shortcut Connection（跳跃连接）**：通过恒等映射（identity mapping）直接把输入 $x$ 加到输出上。

### 网络结构

1. **残差块 (Residual Block)**：

$$
y = F(x, W_i) + x
$$

其中 $F$ 是由 2-3 个卷积层组成的映射。

2. 当输入输出维度不同时，用 $1\times1$ 卷积（projection shortcut）调整维度。

3. **Bottleneck 设计**（用于 50 层及以上的深网络）：采用 1×1 降维 → 3×3 卷积 → 1×1 升维的结构，减少计算量。



然后对比了ResNet和MLP在训练时的loss和$\epsilon$-rank的变化，结果如下

![](/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/ResNet.png)



最后画出不同layers随着迭代的$\epsilon$-rank的变化，变化如下：

<img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/Stair reproduction/photos/different layers D vs R.png" style="zoom:72%;" />



总结：第四章的方法都是以其他文章为灵感或者借用其他文章的方法，特别是与神经元线性独立性有关的方法，来帮助更快提高$\epsilon$-rank

1. 全文分为$\epsilon$-rank引入和staircase现象介绍，理论证明，现有方法应用

2. 文章主要是不再局限于关注训练的loss，而是从代数角度理解网络在每一层生成的特征空间的维数，这也就揭示了网络训练的内在过程：从低秩 → 高秩的逐渐演化。$\epsilon$-rank反映的是**函数空间的表示能力**，因为其是输出函数空间的表示能力的体现，即网络到底在用多少自由度表示数据

3. 为什么$\epsilon$-rank会在训练过程中不断上升呢？（这事实上就是文章发现的现象，并通过这个现象来帮助我们理解深度学习）
   初始时，权重是随机的，很多特征函数高度相关，随着训练，网络权重逐渐调整，激活函数的非线性组合也越来越复杂

4. For deep learning , 直观看到：浅层网络容易过早 saturate（秩不够、表示力有限），而深层网络逐层提升 rank，能捕捉更复杂的结构。ResNet 的 skip connection 避免了梯度消失，并且延缓了特征 rank 的提升，这意味着 ResNet 在训练早期保持了“低秩表示”（即模型先学到简单结构），等到训练更深入时再逐渐提升秩，学习复杂结构。

5. Partition of Unity / Random Feature Method 说明了如何局部地近似复杂函数。ELM、DeepNet、ResNet 则展示了学习型方法如何“自动”找到这些特征。这加深了我们对“为什么深度学习能在高维数据中自动提取低维特征”的理解：它不是 brute force，而是逐步增加秩、逐层细化。



## 第三部分理论解释：

3.1: 这个引理告诉我们，**无论如何排列正交矩阵的列，总存在一个子矩阵的最小奇异值不会太小**

3.2 ：**即使一个函数由 n 个函数组合而成，如果它的 ϵ-rank 只有 p，那么它实际上可以被 p 个函数很好地近似**。
这说明了“有效特征”的数量才是决定表示能力的关键，而不是神经元的数量

3.3 ：
**损失函数的下界由两个因素决定**：

1. dist(u∗,Fp)：真实解与当前函数空间的距离；
2. ϵ：线性独立性的容忍度。

若要降低损失，要么：
减小 dist(u∗,Fp)dist(*u*∗,F*p*)（即提高函数空间的表示能力），也就是**提高 ϵ-rank**；
或者降低 ϵ（提高线性独立性）。

这就**从理论上解释了为什么 ϵ-rank 的增长是损失下降的前提**。

3.4: **定理 3.3 的结论适用于广泛的机器学习与科学计算问题**，包括回归、分类、PINNs 等

3.5: During the training process, to minimize the loss function *L*(*u*), there must be a decrease in dist(u, Fp）implying an increase in the *ϵ*-rank of the neuron functions.

3.6:
如果 ϵ很小，且损失已经很小，但 p 远小于 n，则说明**实际上只需要 p个神经元就足够表示解**，多余的神经元是冗余的。
这为**神经元剪枝**提供了理论依据：我们可以通过监控 ϵ-rank 来判断哪些神经元是重要的，哪些可以剪枝。
