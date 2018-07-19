### 循环神经网络(RNN)
这篇文章很多内容是参考：<http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/>，在这篇文章中，加入了一些新的内容与一些自己的理解。

  循环神经网络(Recurrent Neural Networks，RNNs)已经在众多自然语言处理(Natural Language Processing, NLP)中取得了巨大成功以及广泛应用。但是，目前网上与RNNs有关的学习资料很少，因此该系列便是介绍RNNs的原理以及如何实现。主要分成以下几个部分对RNNs进行介绍： 
1. RNNs的基本介绍以及一些常见的RNNs(本文内容)； 
2. 详细介绍RNNs中一些经常使用的训练算法，如Back Propagation Through Time(BPTT)、Real-time Recurrent Learning(RTRL)、Extended Kalman Filter(EKF)等学习算法，以及梯度消失问题(vanishing gradient problem) 
3. 详细介绍Long Short-Term Memory(LSTM，长短时记忆网络)； 
4. 详细介绍Clockwork RNNs(CW-RNNs，时钟频率驱动循环神经网络)； 
5. 基于Python和Theano对RNNs进行实现，包括一些常见的RNNs模型。

  不同于传统的FNNs(Feed-forward Neural Networks，前向反馈神经网络)，RNNs引入了定向循环，能够处理那些输入之间前后关联的问题。定向循环结构如下图所示： 

![图片](http://img.blog.csdn.net/20150921225125813)

  该tutorial默认读者已经熟悉了基本的神经网络模型。如果不熟悉，可以点击：Implementing A Neural Network From Scratch进行学习。
你是不是在找这个东西
