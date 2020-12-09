---
layout:     post
title:      Pytorch 快速入门
subtitle:   Pytorch基础
date:       2019-10-2
author:     fclef
header-img: img/post-bg-universe.jpg
catalog: 	 true
tags:
​    - AI
​    - Pytorch
---

# Pytorch 快速入门


## Numpy基础

NumPy是Python数值计算最重要的基础包，提供大多数的科学计算方法。其本身并没有多么高级的数据分析功能，但是理解NumPy数组以及面向数学的计算将有助于高效使用Pandas等工具。

**NumPy的主要优点如下：**

1. ndarry，一个具有矢量运算和复杂广播能力的快速且节省空间的多维数组。
2. 可以对整组数据进行快速运算而无需编写循环。
3. 具有用于读写磁盘数据的工具，以及用于操作内存映射文件的工具。
4. 有线性代数、随机数生成以及傅里叶变换等功能。
5. 用于集成由C、C++、Fortran等语言编写的代码的API。

NumPy最重要的一个特点就是其N维数组对象（即ndarray），该对象是一个快速而灵活的大数据集容器。

ndarray的所有元素必须是相同类型的，每个数组都有一个shape（一个表示各维度大小的元组）和dtype（一个用于说明数组数据类型的对象）：

```python
In [1]: import numpy as np
In [2]: data = np.random.randn(2, 3)
In [3]: print(data)
[[ 0.08580635   0.92551568   -0.35612221],
       [-0.17496393   0.89460353   0.67737304]]
In [4]: print(data*10)
[[ 0.85806352   9.25515676   -3.56122214],
       [-1.74963933   8.94603528   6.77373042]]
```

```python
In [7]: print(data.shape)
(2, 3)
In [8]: print(data.dtype)
float64
```

### 创建数组

```python
In [32]: data2 = [ [1, 2, 3, 4], [5, 6, 7, 8] ]
In [33]: arr2 = np.array(data2)
In [34]: print(arr2)
[[1 2 3 4]
 [5 6 7 8]]
In [35]: print(arr2.ndim)
2
```



### 数组运算

```python
In [48]: arr = np.array([ [1, 2, 3], [4, 5, 6] ])
In [49]: print(arr > 3)
[[False False False]
 [ True  True  True]]
In [50]: print(arr**2)
[[ 1  4  9]
 [16 25 36]]
```



### 索引与切片

NumPy数组的索引和切片的用法和Python中数组的操作绝大部分一致，不同的地方在于NumPy数组对切片后的子集进行的赋值操作可以利用广播，而且切片的结果是原数组的子集映射。如果想得到ndarray对象切片的副本而不是映射，就要使用数组的**copy方法**。

![image-20191002161251527](/Users/fclef/Library/Application Support/typora-user-images/image-20191002161251527.png)



### 广播

如果两个数组的维数不同，则元素到元素的操作是不可能的。但是在NumPy中仍然可以对形状不相似的数组进行操作，因为它拥有广播功能。广播遵循的规则如下：

1. 所有用于计算的输入数组都向其中元素个数最多的那一维看齐，元素个数不足的部分通过在前面补1对齐。
2. 输出数组中每一维的长度是输入数组中该维对应的最大长度。
3. 若某个输入数组的某维和输出数组的对应维长度相同或者其长度为1时，这个数组能够用来计算，否则出错。
4. 当输入数组的某个维的长度为1时，沿着此维运算时都用此维上的第一组值。

![image-20191002162836099](/Users/fclef/Library/Application Support/typora-user-images/image-20191002162836099.png)



## Pytorch基础

### Tensor简介

Tensor(张量)是Pytorch中的基本对象，张量只是多维数组/矩阵。PyTorch中的张量类似于numpy的ndarrays，另外，张量也可以在GPU上使用。PyTorch支持各种类型的张量。

Tensor的算术元算与Numpy一样，因此很多numpy相似的运算操作都可以直接迁移过来，另外Tensor与Numpy的array可以进行相互转换：

```python
import numpy
import torch
x = torch.rand(5,3)
y = x.numpy
z = torch.from_numpy(y)
```



### Variable简介

variable是Pytorch另一个基本对象，它可以理解为是对Tensor的一个封装，用于放入计算图中进行前向传播、反向传播和自动求导。在torch中的Variable就是一个存放会变化的值的地理位置。里面的值会不停发生变化，就像一个装鸡蛋的篮子，鸡蛋数会不断发生变化。那谁是里面的鸡蛋呢，自然就是torch的Tensor了。在一个variable中有三个重要属性：data、grad、creator。

- data：表示包含的Tensor数据部分；
- grad：变量传播方向的梯度，这个属性是延迟分配的，而且仅允许进行一次；
- creator：表示创建这个variable的function的引用，该引用用于回溯整个创建链路；

举个例子：

我们定义一个Variable：

```python
import torch
from torch.autograd import Variable # torch 中 Variable 模块
tensor = torch.FloatTensor([[1,2],[3,4]])
# 把鸡蛋放到篮子里, requires_grad是参不参与误差反向传播, 要不要计算梯度
variable = Variable(tensor, requires_grad=True)
 
print(tensor)
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""
 
print(variable)
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
tensor不能反向传播，variable可以反向传播。
"""
```

到目前为止, 我们看不出什么不同, 但是时刻记住, Variable 计算时, 它在背景幕布后面一步步默默地搭建着一个庞大的系统, 叫做计算图, computational graph. 这个图是用来干嘛的? 原来是将所有的计算步骤 (节点) 都连接起来, 最后进行误差反向传递的时候, 一次性将所有 variable 里面的修改幅度 (梯度) 都计算出来, 而 tensor 就没有这个能力。对比tensor和variable的计算:

```python
t_out = torch.mean(tensor*tensor)       # x^2
v_out = torch.mean(variable*variable)   # x^2
print(t_out)
print(v_out)    # 7.5

v_out.backward()    # 模拟 v_out 的误差反向传递
 
# 下面两步看不懂没关系, 只要知道 Variable 是计算图的一部分, 可以用来传递误差就好.
# v_out = 1/4 * sum(variable*variable) 这是计算图中的 v_out 计算步骤
# 针对于 v_out 的梯度就是, d(v_out)/d(variable) = 1/4*2*variable = variable/2
 
print(variable.grad)    # 初始 Variable 的梯度
'''
 0.5000  1.0000
 1.5000  2.0000
'''

```



### CUDA简介

如果你安装了CUDA版本的Pytorch，就可以启用GPU进行一些运算操作了。可以使用`torch.cuda.device`上下文管理器更改所选显卡设备。默认情况下，不支持跨GPU操作，唯一的例外是copy_()。除非启用对等存储器访问，否则对于分布不同设备上的张量，任何启动操作的尝试都将引发错误。



### 第一个Pytorch程序

#### 典线性函数

先创建一些随机训练样本，使其符合经典函数 $$Y = W^TX+b$$分布，再增加一点噪声处理使得样本出现一定的偏差。然后用Pytorch创建一个线性回归的模型，在训练过程中对训练样本进行反向传播，求导后根据指定的损失边界结束训练。



#### 代码实现

```python
#!～/anaconda3/bin/python
# _*_ coding:UTF-8 _*_
from __future__ import print_function
from itertools import count

import numpy as np
import torch
import torch.autograd

import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

random_state = 5000
torch.manual_seed(random_state)
poly_degree = 4
W_target = torch.randn(poly_degree, 1) * 5
b_target = torch.randn(1) * 5


def make_features(x):
    """
    创建一个特征矩阵结构为[x, x^2, x^3, x^4].
    """
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, poly_degree + 1)], 1)


def f(x):
    """近似函数"""
    return x.mm(W_target) + b_target[0]


def poly_desc(W, b):
    """生成多项式描述内容"""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=32):
    """创建类似(x, f(x))批数据"""
    random = torch.from_numpy(np.sort(torch.randn(batch_size)))
    x = make_features(random)
    y = f(x)
    return Variable(x), Variable(y)


# 声明模型
fc = torch.nn.Linear(W_target.size(0), 1)

for batch_idx in count(1):
    # 获取数据
    batch_x, batch_y = get_batch()
    # 重置求导
    fc.zero_grad()

    # 前向传播
    output = F.smooth_l1_loss(fc(batch_x), batch_y)
    loss = output.item()

    # 后向传播
    output.backward()

    # 应用导数
    for param in fc.parameters():
        param.data.add_(-0.1 * param.grad.data)

    # 停止条件
    if loss < 1e-3:
        plt.cla()
        plt.scatter(batch_x.data.numpy()[:, 0], batch_y.data.numpy()[:, 0], label='real curve', color='b')
        plt.plot(batch_x.data.numpy()[:, 0], fc(batch_x).data.numpy()[:, 0], label='fitting curve', color='r')
        plt.title('$Y=W^T*X+b$')
        plt.legend()
        plt.savefig('1.png')
        plt.show()
        break

print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
print('==> Learned function:\t' + poly_desc(fc.weight.data.view(-1), fc.bias.data))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))
```



#### 模型回归结果

![image-20191002183100699](/Users/fclef/Library/Application Support/typora-user-images/image-20191002183100699.png)

#### 损失函数

为什么用smoothL1 loss而不用L1和L2 Loss，主要目的是从两个方面限制梯度的大小：

- 当预测值与真实值差别过大的时候，限制梯度值不过大；
- 当预测值与真实值差别很小时，梯度值足够小。

![image-20191003002144478](/Users/fclef/Library/Application Support/typora-user-images/image-20191003002144478.png)

![image-20191003002156906](/Users/fclef/Library/Application Support/typora-user-images/image-20191003002156906.png)

