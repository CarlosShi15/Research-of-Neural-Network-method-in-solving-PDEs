1. Interface problems  (elliptical interface problem in this paper)

   <img src="/Users/syh/Library/Application Support/typora-user-images/image-20250812160528097.png" alt="image-20250812160528097" style="zoom:50%;" />
   
   

Low regularity of solutions across interfaces coupled with the complex geometry $\to$  accuracy loss (applying numerical methods)

## Ways to mitigate 

**need mesh methods**

1. **Body-fitted meshes** : ensure optimal or near-optimal convergence rates
2. immersed boundary methods $\to$  modifying standard numerical methods on structured grids
3. Interface-capturing method $\begin{cases} explicit & front-tracking\ method \quad \\ implicit & volume\ of\ fluid\ method \quad / \ \ \ level-set\ method \end{cases}$
4. immersed interface method ：adding correction terms to account for jumps in the solution or its derivatives
5. ghost fluid method

**Neural network-based numerical methods** **(mesh free)**

1. Deep Ritz
2. Deep Galerkin
3. Physics-Informed Neural Networks



## 2.2. Generating the hidden-layer neuron location - uniform neuron distribution

下图是文中的Fig2，是数值函数$D_{M}^{\tau}(x)=\frac{1}{M}\sum_{m=1}^{M}\chi_{\{d_m(x)<\tau \}}(x)$ , where $\chi$ is the indicator function

<img src="/Users/syh/Documents/Learning material/Graduates_application/Research experience/Prof. Yang/MultiTransnet/photos/neuron density in balls.png" style="zoom:80%;" />

此均匀分布是Theorem2得到的结果，此结果非常重要，是**transferable**的核心，

Theorem 2 保证了：在球 $B_R(x_c)$ 内，超平面是 **均匀覆盖** 的。好处有：

1. **逼近能力均衡**

   不论函数在哪个位置有变化，网络在各个区域都有足够 neuron 参与近似；

   避免了“有些地方分辨率很高，有些地方几乎没人管”的情况。

2. **更好的泛化性 / 可迁移性**

   因为 neuron 不依赖具体 PDE，只依赖几何区域（通过 Theorem 2 构造），所以同一组 neuron 可以应用在不同 PDE 上；

   **换方程不需要重训 hidden layer**

3. **优化问题变线性**

   在传统神经网络里，hidden layer 位置参数需要训练，这是非凸优化；

   Theorem 2 告诉我们直接用均匀分布的随机参数，就能保证理论上的覆盖性；

   这样只需要解线性 least-squares（求输出层 $\alpha_m$），整个优化问题变得简单、稳定。

4. **误差理论保证**

   文章中公式 $\mathbb{E}[D_M^\tau(x)] = \tau/R$ 本质上给出了一个 **均匀覆盖的统计性质**；

   意味着 neuron 的密度在各处期望一致，所以整体逼近误差可以控制在某个均衡范围。



？
Guassian Random Fields
Grid search

一开始以为方法会研究domain，但是后续用空间中的圆来包住domain，省去了对domain的研究



**variation speed :** 
在特征线法与波动方程中,对于一阶双曲型PDE（如输运方程）或波动方程，解沿特征线传播，其变化速度与特征速度直接相关：
例如，方程 $\frac{\partial u}{\partial t}+c\frac{\partial u}{\partial x}=0$ 的解以速度 $c$ 传播，初始条件的变化以速度 $c$ 沿x轴移动。
此时，"variation speed" 可能指扰动传播的速度（即波速)

**empirical formula-based prediction strategy**    is designed to determine $\gamma$
![image-20250828164303160](/Users/syh/Library/Application Support/typora-user-images/image-20250828164303160.png)



**Multi-transnet 's idea : **
**Using the nonoverlapping domain decomposition to develop multiple transferable neural network method**

这里所谓的multiple transnet事实上就是对切割的subdomain各自用transnet得到solution
solution is $u_{NN}$ in the paper
$u_{NN}(x)=\sum_{m=1}^{M}\alpha_m \sigma(\gamma (a_m^T(x-x_c)+r_m))+\alpha_0$，且这是一个简单的单隐藏层，也就是two-layers
按照Theorem2的分布来确定$\{a_m,r_m\}$的值，分布范围则看domain的具体情况，圆将domain包住，接着需要确定$\gamma$
通过上述empirical formula-based prediction strategy确定$\gamma$，
最后是通过minimize $Loss_{TN}(\alpha)=\lambda_L||L(u_{NN}(x))-f(x)||^2_2+\lambda_B||B(u_{NN}(x))-g(x)||^2_2$来确定$\alpha_m$就得到solution
拼接完就得到solution



在Transnet中，对于$\gamma$，作者是使用高斯随机场（GRFs）生成一系列与目标PDE无关的辅助函数，通过网格搜索找到一个能使网络最好地逼近这些函数的$\gamma$值。**整个过程完全不使用PDE信息**
本文的empirical formula-based prediction strategy是先在一个非常小的网络上用优化策略（如黄金分割搜索），这个黄金分割搜索时我们并不知道$\alpha$的值,分割的是$\gamma$，事实上是计算$min\ \eta(\gamma)$，where $\eta(\gamma)=min_{\alpha} Loss_{TN}(\alpha)$，也就是说每次分割计算的res是当前$\gamma$下通过优化$\alpha$所能达到的最小值



## Illustration : 𝐾 = 2 subdomains ($Ω_1$ and $Ω_2$) and one interface Γ

Multi-TransNet solution is written as : $u^{NN}=u^{NN}_1\chi_{\Omega_1}+u^{NN}_2\chi_{\Omega_2}$

2D loss：
![image-20250831152044250](/Users/syh/Library/Application Support/typora-user-images/image-20250831152044250.png)

Finally transform into a least square problem
![image-20250830144824498](/Users/syh/Library/Application Support/typora-user-images/image-20250830144824498.png)



文章后续又做了一个1D的事例，实验取了一个真解然后代入让两种方法求解。
最后得到的结果显示Multi-transnet完美拟合了真解，而transnet则无法捕捉解的间断性质

**由此可见，此方法的确对原方法有了巨大改进，主要难度应该在解矩阵问题，为保证每个domain拟合足够好，可能矩阵总维度会比较大**



<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250831152230552.png" alt="image-20250831152230552" style="zoom:40%;" />
由此式可知，在得到$\gamma_k$后，$\phi_k$就可以立刻得到

下图是weighting parameters的公式，也就是loss function中每一项前面的factor

<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250901114656443.png" alt="image-20250901114656443" style="zoom:40%;" />

**界面分割考虑的是方程系数是否有突变**

# Numerical Experiment 

**（本文设定的一些结果基本都用实验说话，应该是由于神经网络的难以解释性，没有理论支持）**
**（用实验证明上述方法的设定是比较合理的，合理性主要体现在得到结果的误差较小 ）**

## 5.1. Ablation studies

### 5.1.1 TransNet

**Benefits of translating and scaling hidden-layer neurons**：

Fig7处理的并不是非常完美吧，因为not translating比translating还要好一些，可能和覆盖区域面积有关吧

区域划分是基于问题几何的，是显式的

**Effectiveness of the empirical formula-based prediction strategy：**
**training loss-based optimization** VS **empirical formula-based prediction strategy** 
前者是在minimize the loss function to get $\gamma$ ，文中的algorithm 1 即是此optimization
后者是在前者基础上add empirical formula to find $\gamma^*$

### 5.1.2 Multi-TransNet

**Effect of globally uniform neuron distribution**：
error更小

**Effectiveness of the empirical formula-based prediction strategy**：
如果两种方法精度差不多的话，那empirical肯定效率更高，因为只需要计算400neurons时的即可，后续可用经验公式。如果没有经验公式的话，那么每一次不同的neuron数都要解一次很大的least square

**Impact of the weighting parameters in the loss function**：Normalizingthe least squares augmented matrix is the best choice



## 5.2. Applications of the Multi-TransNet to typical elliptic interface problems

### 5.2.1. A 2D Stokes interface problem with a circular interface

compare with RFM，本方法更优

### 5.2.2. A 2D diffusion interface problem with multiple interfaces

Multi-TransNet remarkably outperforms the LRNN

### 5.2.3 A 3D elasticity interface problem with an ellipsoidal interface

三个居中相交的平面是为了克服3D可视化的挑战，将体积数据投影到二维平面上以便展示
Multi-TransNet easily and significantly beats the NIPFEM

### 5.2.4. A 3D diffusion interface problem with a convoluted immersed interface

文章中提到了three coordinate planes (xy,yz,zx)，**XY-plane**是一个由方程 Z = 0定义的平面，其余两个同理

compare Multi-transnet with **cusp-capturing PINN method**

caseI: spatially varying diffusion coefficient
caseII: piecewise constant diffusion coefficient (i.e. use the contrast $\dfrac{\beta_1}{\beta_2}$)

**Both 5.2.3 and 5.2.4 caseII use parameter contrast to determine the parameters and find fluctuations for $L_{\infty}$ when M=4000 and 2000 respectively** 



## Some potential future work includes:

(1) improving the assembling and solving efficiency of the resulting least squares problem; 
即求解 min ||Fα - T||² 的计算效率

(2) developing more effective approachesfor generating the hidden-layer neurons based on specific target domains; 
文中是用球包住domain，然后$\{a_m\}$ are i.i.d. and uniformly distributed on the d-dimensional unit sphere,$\{r_m\}$ are i.i.d. and uniformly distributed in [0, R]
然后在总神经元个数确定的情况下，根据覆盖各个domain的圆的radius来分配$M_i$ ，($i.e.$ $\dfrac{M_i}{R_i}=\dfrac{M_j}{R_j}$)这一点也就保证了globally uniform distribution分配神经元

(3) extending the present method to dynamic interface problems.



Idea：
文章也可以引入$\epsilon$-rank作为评价指标，看multi-transnet方法是否优秀，如果效果不明显可以做哪些改进
但是问题是按照本文experiment的迭代逻辑是将神经元个数作为iteration，因此比较起来并不方便
因此我们还是要按照staircase的逻辑来计算某个参数迭代时对$\epsilon$-rank的影响，比如$\gamma$，但这样就没法用empirical formula
所以总体来说两篇文章逻辑有点难以粘合





Example K=2 subdomains   1 dimension

$u^{NN}=u^{NN}_1\chi_{\Omega_1}+u^{NN}_2\chi_{\Omega_2}$

$u^{NN}_k=\alpha^T_k\phi_k$       where    $\alpha_k=(\alpha_0^{(k)},\alpha_1^{(k)},\cdots,\alpha_{M_k}^{(k)})^T$   $\phi_k=(\phi_0^{(k)},\phi_1^{(k)},\cdots,\phi_{M_k}^{(k)})^T$     $k=1,2$

$\phi_0^{(k)}=1$    $\phi_m^{(k)}=\sigma(\gamma_k((x-x_c^{(k)})^Ta_m^{(k)}+r_m^{(k)}))$   where $m=1,2,\cdots,M_k$  , $\sigma(x)=tanh(x)$ 

$x_{c_1}, R_1 = 0.125, 0.15$       $x_{c_2}, R_2 = 0.625, 0.40$    $M1=M2=5$   for $\Omega_1=(0,1/4)$  $\Omega_2=(1/4,1)$        (specially Set)

$a_m^{(k)},r_m^{(k)}$ are uniformly distributed in the sphere and $[0,R^{(k)}]$ respectively 

Here $\gamma_1,\gamma_2$  is get by **golden-section search** since we have $\gamma_2=\gamma_1\dfrac{R_1}{R_2}\dfrac{M_2}{M_1}$  ，we randomly choose several points in each required domain in expression of $Loss_{MT}(\alpha)$

so we get $\gamma_1$ by $min_{\gamma_1} \ \eta(\gamma_1,\gamma_2=\gamma_1\dfrac{R_1}{R_2}\dfrac{M_2}{M_1}) $     then we can get $\gamma_2$

Here $\eta(\gamma_1,\gamma_2)=min_{\alpha_1,\alpha_2} Loss_{MT}(\alpha)$

<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250906155415613.png" alt="image-20250906155415613" style="zoom:40%;" />

the expression is corresponding to the problem 

<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250812160528097.png" alt="image-20250812160528097" style="zoom:40%;" />

Finally get alpha by <img src="/Users/syh/Library/Application Support/typora-user-images/image-20250906155909739.png" alt="image-20250906155909739" style="zoom:40%;" />

$i.e.$ For the problem given is   (include：two interior、boundary conditions、 $[u]=h_1$ and $[\beta u'] = h_2$） 

Then we get all the parameters for $u_{NN}$





