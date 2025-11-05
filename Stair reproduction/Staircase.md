# Effective Rank (ER):

measure the linear independence of the basis function represented by neurons in the final hidden layer

ER increase as a stair-like pattern during training and **Loss function L** decrease

ER $\propto$ $\dfrac{1}{L}$       Lower bound of loss function decrease with increasing effective rank

$\Rightarrow$  achieve rapid descent of loss function by promote the growth of ER



# Reference:

[11,10]   ReLU-activated deep neural networks can reproduce all linear finite element functions

[23, 24, 30]   provide **algorithms** to combine classical finite element methods with neural networks.

[3] A detailed analysis of the coefficient matrices associated with the random features, particularly in terms of the distribution of their singular values, has been provided 

[40] The decay rate of the **eigenvalues of the Gram matrix** of a two-layer neural network with ReLU activation under general initialization has been analyzed （这个在function fitting中应该要用）

[12] Residual Network

[9] Initialization methods





Visualization-based methods: 例如降维可视化，也就是将高维数据投影到低维观察聚类情况，这种从低维分析高维的方式还有流形学习

Neural tangent kernel: 将Infinitely Wide Neural Networks的训练动力学与Kernel Methods联系起来

Frequency principle: 神经网络在训练过程中，会优先快速学习目标函数中的低频成分（平滑、整体的趋势），然后才缓慢地学习高频成分（细节、起伏、噪声）
