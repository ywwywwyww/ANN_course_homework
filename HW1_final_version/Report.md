# MNIST Digit Classification with MLP

袁无为 计预0

## Abstract

　　图像识别一直是深度学习中的一个基础又重要的内容。我们对手写数字识别和分类进行了一些研究，搭建了 MLP 模型并使用 MNIST 数据集对我们的模型进行了一些测试。

## 1. Introduction

　　手写数字识别是一个非常简单的问题。对于给定的一张 $28\times 28$ 个像素的只包含一个数字的灰度图，只需要识别出这个图像对应的数字。而且这个数字往往会出现在图像的中央。

　　把一个图像输入到 MLP 中，会得到预测出的这个图像分别是每个数字的概率。

　　下图展示了两个手写数字的图像、对应的数字以及 MLP 预测的结果（一个预测正确和一个预测错误的例子）：



## 2. Approach

　　我使用的 MLP 模型主要包含以下几部分内容：

### 2.1 Basic Structure

　　我搭建了几个包含一个隐藏层的和几个包含两个隐藏层的模型，它们的结构如下：

![包含一个隐藏层的 MLP](https://www.researchgate.net/profile/Rosline_Hassan/publication/260321700/figure/fig1/AS:296985614667776@1447818296312/Structure-of-a-one-hidden-layer-MLP-Network.png) ![Image result for mlp with two hidden layer](https://www.researchgate.net/profile/L_Ekonomou/publication/236900080/figure/fig1/AS:299287272542214@1448367054951/Multilayer-perceptron-MLP-with-two-hidden-layers.png)

　　所有网络的输入层节点个数为 $784$，输出层节点个数为 $10$。边为全连接的。没有 Dropout。

　　对于有一个隐藏层的网络，隐藏层的节点数为 $100$。

　　对于有两个隐藏层的网络，第一个隐藏层的节点个数为 $200$，第二个隐藏层的节点数为 $100$。

### 2.2 Activation Function

　　我选择了两个激活函数：
$$
\begin{align}
\text{ReLU}(x)&=\begin{cases}x&,x>0\\
0&,\text{otherwise}
\end{cases}\\
\text{Sigmoid}(x)&=\frac{1}{1-e^{-x}}
\end{align}
$$
　　它们的导数分别为：
$$
\begin{align}
\text{ReLU}'(x)&=\begin{cases}1&,x>0\\
0&,x<0
\end{cases}\\
\text{Sigmoid}'(x)&=\frac{e^{-x}}{{(1+e^{-x})}^2}=\text{Sigmoid}(x)(1-\text{Sigmoid}(x))
\end{align}
$$

### 2.3 Loss Function

　　我选择了两个误差函数：

　　　EuclieanLoss(MeanSquareError)：对于给定的预测结果 $y^{(n)}$ 和实际的结果（人工标注的结果） $t^{(n)}$，误差为：
$$
E=\frac{1}{2N}\sum_{n=1}^N\lvert \lvert t^{(n)}-y^{(n)}\rvert \rvert _2^2
$$
，其中 $N$ 为 batch size。它的导数为
$$
\frac{\partial E}{\partial y^{(n)}}=\frac{1}{N}(y^{(n)}-t^{(n)})
$$
　　　SoftmaxCrossEntropy：对于给定的预测结果 $y(x)^{(n)}$ 和实际的结果 $t(x)^{(n)}$，误差 $E$ 为：

$$
E=\frac{1}{N}\sum_{n=1}^NE^{(n)}\\
E^{(n)}=-\sum_{k=1}^Kt_k^{(n)}\ln h_k^{(n)}\\
h_k^{(n)}=\frac{\exp(y_k^{(n)})}{\sum_{j=1}^K\exp(y_j^{(n)})}
$$
 　　其中 $K$ 是分类的类别数。它的导数为：
$$
\frac{\partial E}{\partial y_k^{(n)}}=\frac{1}{N}(h_k^{(n)}-t_k^{(n)})
$$

### 2.4 Parameters

　　我使用了 Xavier 初始化方法，权值矩阵初始化为 $\sigma^2=\frac{1}{n_i}$ 的正态分布的随机变量。其中 $n_i$ 为前一层的节点个数。

　　偏置矩阵初始化为 $0$。

### 2.5 Hyperparameters

　　我尝试了多种不同的组合，最终在对比中使用的参数为：learning rate $=0.1$，momentum $=0$，weight decay $=0$，batch size $=100$。这组参数对于某一个特定的模型并不一定是最优的，只是对于所有模型都有较好的表现。我还会对比这些参数对特定模型的影响。

### 2.6 Optimizer

　　使用了经典的 SGD 方法。

### 2.7 Normalization

　　在输入层前先对数据做一次 Normalization，即 $x_i'=\frac{x_i-\overline x}{\sigma}$ 其中 $\overline x$ 为 $x_i$ 的平均值，$\sigma$ 为 $x_i$ 的标准差。这样做能把 $x_i$ 调整为平均值为 $0$，方差为 $1$ 的分布的变量。能够减少微小扰动带来的影响。

## 3. Experiments

### 3.1 Datasets

　　使用了经典的 MNIST 手写数字数据集。

### 3.2 Implementation

#### 不同结构的网络之间的对比

　　搭建不同的网络，使用相同的超参数进行测试。

#### 不同的超参数之间的对比

　　搭建一个网络，使用不同的超参数进行测试。

#### 不同的初始化参数之间的对比

　　搭建一个网络，使用不同的参数初始化权重矩阵进行测试。

#### Normalization 的影响

　　搭建两个网络，其中一个网络在输入层对数据做一遍 normalization。

### 3.3 Quantitative Results

#### 不同结构的网络之间的对比

　　分别搭建了八个不同的网络，使用了相同的参数进行对比（其中误差函数为 SoftmaxCorssEntropy 的网络中输出层没有激活函数）。使用的参数为：learning rate $=0.1$，momentum $=0$，weight decay $=0$，batch size $=100$。结果如下：

(layer=1/layer=2),(sigmoid/relu),(mse/softmaxcrossentropy)

#### 不同超参数之间的对比

#### 不同的初始化参数之间的对比

(std=sqrt(1/n),std=sqrt(2/n)),(sigmoid/relu)

softmaxcrossentropy

#### Normalization

(has/hasn't normalization)

## 4. Conclusion