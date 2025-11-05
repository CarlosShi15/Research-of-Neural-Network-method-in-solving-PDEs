Transnet：its key point is to con struct a transferable neural feature space based on re-parameterization of the hidden layer neurons
 and approximation properties without using information from PDEs

re-parameterization即添加了gamma等，approximation properties指的是transnet可以被视为一个万能逼近器，只需确定P_NN还有均匀分布的超平面以及参数am和rm



**MAE-Transnet : utilizing the theory of matched asymptotic expansions**

文章以transnet为基础，首先发现该方法同样不能很好的解决single perturbation问题，因此考虑改进方法，故而加入了MAE

benchmark problem：评估算法性能的问题

single perturbation : 小参数乘以最高阶导数时产生的特殊数学现象

Problem that focused by the paper :
<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250902174125954.png" alt="image-20250902174125954" style="zoom:50%;" />



How to determine the location of the boundary layer:
The failure of the outer solution to satisfy both boundary conditions tells you where the boundary layer must be 
($i.e.$ check the boundary condition)



Example：
<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250903153934609.png" alt="image-20250903153934609" style="zoom:50%;" />

小参数$\epsilon$乘在最高阶的gradient上，此问题可以看成一个对流-扩散问题，二阶是扩散项，一阶是对流项，0阶是反应项

**核心矛盾**：边界条件要求在 x=0和 x=1 的函数值分别是 0 和 1。但当 ε非常小时，扩散效应很弱，方程主要由对流项$\dfrac{du}{dx}$主导。一个一阶方程通常只能满足一个边界条件。这就导致了冲突，系统为了“满足”两个边界条件，会在某个地方形成一个变化极其剧烈的薄层，即**边界层（Boundary Layer）**

**为什么边界层在x=0处**：我们有方向性准则：
想象你是一个流体粒子，从边界向域内移动。对流项的方向决定了信息的传播方向，本问题中系数为正。这意味着信息是从左向右传播的（正x方向）
因此在上游（Upstream）边界（x=0），边界条件 u(0)=0无法通过向右的流动影响到内部，因为它处在“信息源”的反方向。为了满足这个“被孤立”的边界条件，系统不得不在 x=0附近形成一个极其陡峭的梯度（即边界层），通过强大的扩散效应（尽管ε很小，但二阶导数 $\dfrac{d^2u}{dx^2}$可以非常大）来“强行”满足 u(0)=0

**外解是描述边界层之外的平滑行为的解**。边界层是一个薄薄的层，在内部解剧烈变化。在边界层之外，解的变化不剧烈，因此其二阶导数项$\epsilon \dfrac{d^2u}{dx^2}$的值很小, 外解在 x=1附近是有效的，所以我们使用 x=1处的边界条件来确定外解

通过以上，我们就基本确定了外解，然后我们就要将中心放在内解上。内解就是边界层内部的解。

于是我们现在要确定A和B，就需要两个边界条件。其一在原方程中给出了，也就是u(0)=0，其二就需要用到**matching principle**

最后我们就得到了复合解 $u_c(x)$
$u^c(x)=u^i(x)+u^o(x)-overlap=u^i(x)+u^o(x)-e=e^{1-x}-e^{1-\dfrac{x}{\epsilon}}$



疑问：
<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250903174903306.png" alt="image-20250903174903306" style="zoom:50%;" />

如果这里代入$\dfrac{1}{\epsilon}$，就相当于$u^i(1)$，相当于内解在1和外解在0相等。有点不知道为什么



the hidden-layer neurons are uniformly distributed within a small spherical region covering only the scaled boundary layer region (such as (0*,* 1) *⊂* $Ω^ζ$ in this case) rather than the entire scaled computational domain $Ω^ζ$

这样做是**为了将有限的神经网络计算能力（神经元）“集中火力”用在最需要的地方——边界层内部**
从内解 $u^i(ζ)=e(1−e^{−ζ})$可以看出，它的变化几乎全部集中在 ζ 很小的区间内，姑且视为【0，5】,在这个区域之外，函数是平坦的。如果平均分在放大域【0，1000】内，结果是**99.5%的神经元被浪费在了函数为常数的区域**，只有极少数的神经元落在了真正需要拟合函数剧烈变化的 [0,5] 区间。

一个形如tanh神经元，它不是一个“局部”的基函数（像有限元基函数那样只在某个小区域非零)，它是一个**全局函数**。在越过其过渡层（transition layer）后，它的输出会饱和到一个常数值（对于tanh是1或-1）。因此，一个部署在 [0,5] 区间的神经元，不仅可以精确描述该区域内的变化，其饱和后的常数值也能自然地描述 [5,1000]区域的平坦解。



3.1.1 P10
<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250903193101572.png" alt="image-20250903193101572" style="zoom:50%;" />

？如果$m\neq n$怎么办

boundary layer该怎么找

**3.2.2. The case of coupled boundary layers**

在二维空间内有两个边界层，会导致有一个耦合区域

方程从22变到23，24是因为做了新坐标引入后会出现$\frac{1}{\epsilon}$，因此能分别消掉一些内容

耦合区域的边界条件是和$u^c_{NN}$对应



在MAE-transnet中，我们的loss function
$Loss=||L(u(x))-L(u_{NN}(x))||^2_2+||B(u(x))-B(u_{NN}(x))||^2_2$



## 第三节的后三个示例问题：

3.1.2
<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250908225213622.png" alt="image-20250908225213622" style="zoom:50%;" />

3.2.1
<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250908225134446.png" alt="image-20250908225134446" style="zoom:50%;" />
一个boundary layer里的边界和一个matcheding principle 得到的边界

3.2.2
<img src="/Users/syh/Library/Application Support/typora-user-images/image-20250908225049051.png" alt="image-20250908225049051" style="zoom:50%;" />

像本问题，如果boundary layers是通过outer solution和边界条件的对应性来确定的，那么我们就需要先解出outer solution然后来找到boundary layers的位置，然后再考虑边界条件





局限性：
高位问题，含时间问题，解比较复杂的，有奇异性



4.1 Testcase1

MAE-TransNet solution $u^c_{NN}$ achieve the best accuracy with just a total of $M^i +M^0$ = 20 neurons
? 为什么后面不再降了

1. PINN and TransNet fails to solve this singular perturbation problem
2. as BL-PINN does, the solution accuracy of MAE-TransNet gets improved when *ε* decreases, which is consistent with the theory of matched asymptotic expansions
3. compared to BL-PINN, our MAE-TransNet requires significantly fewer neurons and less running times while achieves more accurate solutions.



**4.2. Test case 2: 1D linear problem with two boundary layers of the same thickness**

**4.3. Test case 3: 1D linear problem with two boundary layers of different thicknesses**

为什么thickness会不一样
这里same和different不是指几次试验间的$\epsilon$不一样，而是指不同的layers之间的$\delta(\epsilon)$不一样

**4.4. Test case 4: 1D nonlinear problem with a single boundary layer**

For the nonlinear singular perturbation problem with different values of ε, MAE-TransNet can still achieve high accuracy while using the same hidden layer parameters （shows the transferability）

**4.5. Test case 5: 2D Couette flow problem**

We observe that both the MAE-TransNet and BL-PINN solutions closely match the reference solution, and the MAE-TransNet solution converges noticeably closer to the reference solution on the bottom wall within the boundary layer (i.e., *y* = 0)

**4.6. Test case 6: 2D coupled boundary layers problem**

It is observed that the two solutions exhibit almost no differences either across the entire domain Ω or in the coupled region Ω*ii* .（MAE-TransNet solution and the exact solution）

**4.7. Test case 7: 3D Burgers vortex problem**

MAE-TransNet effectively captures the global features of the Burgers vortex



method provides a promising tool for solving singular perturbation problems with boundary layers. There are several potential related topics that will be studied in the future work. Although various numerical results illustrate that the proposed method converges well and has a good accuracy, exploring rigorously theoretical convergence analysis and error estimates is still an interesting topic. It is also worthy of further investigation into more challenging singular perturbation problems involving moving thin layers [25, 47] or turning points [48], as well as coordinate perturbation problems discussed in







