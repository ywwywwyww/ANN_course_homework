# README

## How to reproduce my results

　　运行 run_mlp.py 即可开始训练网络并在 codes\log 目录下保存数据以及简单的图像。

　　运行 draw2.py 即可根据保存的数据绘制复杂一点的图像。

## draw.py

　　新增加了几个类和函数，用于绘制图像和保存数据

## draw2.py

　　新增加了几个类和函数， 用于绘制图像

## layers.py

### class Relu

　　实现了 ReLU 函数、其导数、前向传播以及反向传播

### class Sigmoid

　　实现了 Sigmoid 函数、其导数、前向传播以及反向传播

### class Linear

　　实现了前向传播和反向传播

### class Normalization

　　新增了归一化的前向传播

　　由于是在第一层就没有实现也没必要实现反向传播

## loss.py

### EuclideanLoss

　　实现了前向传播和反向传播

### SoftmaxCrossEntropyLoss

　　实现了前向传播和反向传播

## run_mlp.py

　　增加了几个模型

　　修改了超参数

　　添加了绘制图表的代码

　　添加了计时相关的代码

## solve_net.py

　　添加了绘制图表、保存数据的代码

