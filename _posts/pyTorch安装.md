---

layout:     post

title:      深度学习介绍

subtitle:   Pytorch介绍安装

date:       2019-10-2

author:     fclef

header-img: img/post-bg-universe.jpg

catalog: true

tags:

​    - AI

​    - Pytorch

​    - Notes

---

# 深度学习介绍

## 人工智能、机器学习与深度学习

**人工智能**：指由人制造出来的机器所表现出来的智能（维基百科）。通常广义上的人工智能是指让机器拥有完全的甚至超越人类的智能（强AI）；而现代计算机科学当中的研究更多聚焦在研究如何让计算机模拟人类的智能来实现某些特定的依赖人类智能才能实现的任务(弱AI)。

**发展历程：**

![image-20191002112123236](/Users/fclef/Library/Application Support/typora-user-images/image-20191002112123236.png)

![image-20191002112304513](/Users/fclef/Library/Application Support/typora-user-images/image-20191002112304513.png)

## 深度学习工具

![image-20191002113719900](/Users/fclef/Library/Application Support/typora-user-images/image-20191002113719900.png)

### Theano

最初诞生于蒙特利尔大学 LISA 实验室，于2008年开始开发，是第一个有较大影响力的Python深度学习框架。**由于Theano已经停止开发，不建议作为研究工具继续学习。**



### Tensorflow

2015年11月10日，Google宣布推出全新的机器学习开源工具TensorFlow。 ensorFlow在很大程度上可以看作Theano的后继者，不仅因为它们有很大一批共同的开发者，而且它们还拥有相近的设计理念，都是基于计算图实现自动微分系统。TensorFlow 使用数据流图进行数值计算，图中的节点代表数学运算， 而图中的边则代表在这些节点之间传递的多维数组（张量）。

TensorFlow编程接口支持Python和C++。随着1.0版本的公布，Java、Go、R和Haskell API的alpha版本也被支持。作为**不完美但最流行的深度学习框架，社区强大，适合生产环境。**

不完美的地方举三个栗子：

- 频繁变动的接口
- 复杂的系统设计
- 接口设计晦涩难懂



### Caffe/Caffe2

Caffe的全称是Convolutional Architecture for Fast Feature Embedding，它是一个清晰、高效的深度学习框架，核心语言是C++，它支持命令行、Python和MATLAB接口，既可以在CPU上运行，也可以在GPU上运行。

Caffe2的设计追求轻量级，在保有扩展性和高性能的同时，Caffe2 也强调了便携性。Caffe2 从一开始就以性能、扩展、移动端部署作为主要设计目标。Caffe2 的核心 C++ 库能提供速度和便携性，而其 Python 和 C++ API 使用户可以轻松地在 Linux、Windows、iOS、Android ，甚至 Raspberry Pi 和 NVIDIA Tegra 上进行原型设计、训练和部署。Caffe2仍然是一个不太成熟的框架，**文档不够完善，但性能优异，几乎全平台支持（Caffe2），适合生产环境。**

另外作者就是前段时间离开脸书加入阿里的贾清扬。



### CNTK

015年8月，微软公司在CodePlex上宣布由微软研究院开发的计算网络工具集CNTK将开源。5个月后，2016年1月25日，微软公司在他们的GitHub仓库上正式开源了CNTK。早在2014年，在微软公司内部，黄学东博士和他的团队正在对计算机能够理解语音的能力进行改进，但当时使用的工具显然拖慢了他们的进度。于是，一组由志愿者组成的开发团队构想设计了他们自己的解决方案，最终诞生了CNTK。

根据微软开发者的描述，CNTK的性能比Caffe、Theano、TensoFlow等主流工具都要强。CNTK支持CPU和GPU模式，和TensorFlow/Theano一样，它把神经网络描述成一个计算图的结构，叶子节点代表输入或者网络参数，其他节点代表计算步骤。CNTK 是一个非常强大的命令行系统，可以创建神经网络预测系统。CNTK 最初是出于在 Microsoft 内部使用的目的而开发的，一开始甚至没有Python接口，而是使用了一种几乎没什么人用的语言开发的，而且文档有些晦涩难懂，推广不是很给力，导致现在用户比较少。但就框架本身的质量而言，CNTK表现得比较均衡，没有明显的短板，并且在语音领域效果比较突出。
**社区不够活跃，但是性能突出，擅长语音方面的相关研究。**



### Keras

Keras是一个高层神经网络API，由纯Python编写而成并使用TensorFlow、Theano及CNTK作为后端。Keras为支持快速实验而生，能够把想法迅速转换为结果。Keras应该是深度学习框架之中最容易上手的一个，它提供了一致而简洁的API， 能够极大地减少一般应用下用户的工作量，避免用户重复造轮子。

严格意义上讲，Keras并不能称为一个深度学习框架，它更像一个深度学习接口，它构建于第三方框架之上。Keras的缺点很明显：过度封装导致丧失灵活性。Keras最初作为Theano的高级API而诞生，后来增加了TensorFlow和CNTK作为后端。为了屏蔽后端的差异性，提供一致的用户接口，Keras做了层层封装，导致用户在新增操作或是获取底层的数据信息时过于困难。同时，过度封装也使得Keras的程序过于缓慢，许多BUG都隐藏于封装之中，在绝大多数场景下，Keras是本文介绍的所有框架中最慢的一个。

学习Keras十分容易，但是很快就会遇到瓶颈，因为它缺少灵活性。另外，在使用Keras的大多数时间里，用户主要是在调用接口，很难真正学习到深度学习的内容。

**入门最简单，但是不够灵活，使用受限。**



### PyTorch

2017年1月，Facebook人工智能研究院（FAIR）团队在GitHub上开源了PyTorch，并迅速占领GitHub热度榜榜首。PyTorch的历史可追溯到2002年就诞生于纽约大学的Torch。Torch使用了一种不是很大众的语言Lua作为接口。Lua简洁高效，但由于其过于小众，用的人不是很多。在2017年，Torch的幕后团队推出了PyTorch。PyTorch不是简单地封装Lua Torch提供Python接口，而是对Tensor之上的所有模块进行了重构，并新增了最先进的自动求导系统，成为当下最流行的动态图框架。

- 简洁：PyTorch的设计追求最少的封装，尽量避免重复造轮子。不像TensorFlow中充斥着session、graph、operation、name_scope、variable、tensor、layer等全新的概念，PyTorch的设计遵循tensor→variable(autograd)→nn.Module 三个由低到高的抽象层次，分别代表高维数组（张量）、自动求导（变量）和神经网络（层/模块），而且这三个抽象之间联系紧密，可以同时进行修改和操作。 
  简洁的设计带来的另外一个好处就是代码易于理解。PyTorch的源码只有TensorFlow的十分之一左右，更少的抽象、更直观的设计使得PyTorch的源码十分易于阅读。
- 速度：PyTorch的灵活性不以速度为代价，在许多评测中，PyTorch的速度表现胜过TensorFlow和Keras等框架 。框架的运行速度和程序员的编码水平有极大关系，但同样的算法，使用PyTorch实现的那个更有可能快过用其他框架实现的。
- 易用：PyTorch是所有的框架中面向对象设计的最优雅的一个。PyTorch的面向对象的接口设计来源于Torch，而Torch的接口设计以灵活易用而著称，Keras作者最初就是受Torch的启发才开发了Keras。PyTorch继承了Torch的衣钵，尤其是API的设计和模块的接口都与Torch高度一致。PyTorch的设计最符合人们的思维，它让用户尽可能地专注于实现自己的想法，即所思即所得，不需要考虑太多关于框架本身的束缚。
- 活跃的社区：PyTorch提供了完整的文档，循序渐进的指南，作者亲自维护的论坛 供用户交流和求教问题。Facebook 人工智能研究院对PyTorch提供了强力支持，作为当今排名前三的深度学习研究机构，FAIR的支持足以确保PyTorch获得持续的开发更新，不至于像许多由个人开发的框架那样昙花一现。
- 支持ONNX格式：话不多说，使得Pytorch的深度学习应用易于部署生产环境和大规模部署，比如你可以用Pytorch作研究，然后用ONNX转换为caffe2部署到生产环境。



## Pytorch安装

### Anaconda安装

[anaconda安装：macos](https://www.jianshu.com/p/dbf20c6792fe)

[anaconda介绍以及安装教程：linux and windows](https://www.jianshu.com/p/742dc4d8f4c5)

Anaconda，这是一个基于Python的数据处理和科学计算平台，它已经内置了很多非常有用的第三方库，安装上Anaconda，就相当于把数十个第三方模块自动安装好了，非常简单易用。

### Pytorch安装

进官网，选择对应的版本配置，然后使用官网提供的命令在本机的终端执行。

![image-20191002151951367](/Users/fclef/Library/Application Support/typora-user-images/image-20191002151951367.png)



这里有一点需要注意，很多人在使用`conda install pytorch torchvision -c pytorch`出现**CondaHttpError**的错误，这里是因为`-c pytorch`使得安装源并没有使用我们指定的清华源，所以这里可以加一句清华源的配置并去掉`-c pytorch`，执行如下两个命令：

```python
// conda配置清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
// 安装pytorch
conda install pytorch torchvision 
```

做到这一步按理说环境和框架都已经安装好了，可以进入python环境执行`import torch`，如果没有问题则说明Pytorch安装成功。







