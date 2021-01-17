# Learning to Learn with quantum neural networks via classical neural network (QOSF Mentorship Program)

This repository implements the architecture proposed by Verdon et al. in the paper *Learning to learn with quantum neural networks via classical neural networks* [[1]](#1), using **PennyLane** [[2]](#2) and **TensorFlow** [[3]](#3).  

PennyLane is a cross-platform Python library for differentiable programming of quantum computers, which, in this tutorial will be used to create quantum circuits and interface them with TensorFlow, an open-source platform for machine learning, here used to build a custom model of a Recurrent Neural Network (RNN). 

The idea proposed in the paper is to train a classical Recurrent Neural Network (RNN) to assist the optimization of variational quantum circuits, which are known to suffer from trainability issues related to random initialization of parameters. Using this setup, authors in [[1]](#1) showed that it is possible rapidly find approximate optima in the parameter landscape for several classes of quantum variational algorithms.  

In this notebook, we focus on applying such hybrid quantum-classical procedure to leran a good initialization strategy for solving the MaxCut problem using QAOA.   

> The code was developed under the Quantum Open Source Foundation (QOSF) Mentorship Program. More on: https://qosf.org/

---

#### References
<a id="1">[1]</a> 
Verdon G., Broughton M., McClean J. R., Sung K. J., Babbush R., Jiang Z., Neven H. and Mohseni M. (2019),  
Learning to learn with quantum neural networks via classical neural networks, [arXiv:1907.05415](https://arxiv.org/abs/1907.05415).

<a id="2">[2]</a> 
https://pennylane.ai/

<a id="3">[3]</a> 
https://www.tensorflow.org/

