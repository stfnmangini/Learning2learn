# Learning to learn with quantum neural networks via classical neural network (w/ PennyLane & TensorFlow)  

<p align="center">
  <a href="https://colab.research.google.com/github/stfnmangini/Learning2learn/blob/main/Learning2Learn.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg">
  </a> 
</p>
  
<img src="/thumbnail3.png" width="300px" align="right">

> This project was created as part of the Quantum Open Source Foundation (QOSF) Mentorship Program. More on: https://qosf.org/  

This repository implements the architecture proposed by Verdon et al. in the paper *Learning to learn with quantum neural networks via classical neural networks* [[1]](#1), using **PennyLane** [[2]](#2) and **TensorFlow** [[3]](#3).  

### Project desctiption
---
Variational Quantum Algorithms (VQAs) are powerful tools which promise to take full advantage of near term quantum computers. However, these algorithms suffer from optimization issues related to random initialization of the parameters. Using PennyLane and Tensorflow, this repository implements the architecture proposed by Verdon et al. in *Learning to learn with quantum neural networks via classical neural networks*, which leverage a classical Recurrent Neural Network (RNN) to assist the optimization of variational quantum algorithms by learning an efficient parameter initialization heuristics to ensure rapid training and convergence.  

More in detail, by means of an hybrid quantum-classical recurrent setup, a Long-Short Term Memory (LSTM) is used as a black-box controller to initialize the parameters of a variational quantum circuit. In particular, in this notebook we focus our attention on the optimization of a QAOA quantum circuit to solve the MaxCut problem. An outline of the architecutre is the following, and you can find a careful explanation in the notebook. 

![RNN scheme](/HybridLSTM.png)  

> If you need a quick recap on how to use QAOA to solve MaxCut, check out this great demo: https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut.html

### Required packages  
---
The Jupyter Notebook is written in `Python`, and the following packages are needed to run the code:  
- `PennyLane`:  a cross-platform `Python` library for differentiable programming of quantum computers, which in this tutorial will be used to create quantum circuits and interface them with  
- `TensorFlow`: an open-source platform for machine learning, here used to build a custom model of a Recurrent Neural Network (RNN)
- `NetworkX`: a `Python` package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks  
- `Numpy`, `Matplotlib`: standard libraries for array manipulation and plotting   

#### References  
---
<a id="1">[1]</a> 
Verdon G., Broughton M., McClean J. R., Sung K. J., Babbush R., Jiang Z., Neven H. and Mohseni M. (2019),  
Learning to learn with quantum neural networks via classical neural networks, [arXiv:1907.05415](https://arxiv.org/abs/1907.05415).

<a id="2">[2]</a> 
https://pennylane.ai/

<a id="3">[3]</a> 
https://www.tensorflow.org/
