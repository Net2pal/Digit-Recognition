# Digit Recognition



Overview
============
This project helps train a classifier to recognize handwritten character images [MNIST digits](http://yann.lecun.com/exdb/mnist/). It uses logistic regression as it's model and trains on a small MNIST dataset before testing the trained model on it. You can view the constructed data flow graph using Tensorboard in your browser after training.
<p>
<img src="https://camo.githubusercontent.com/d440ac2eee1cb3ea33340a2c5f6f15a0878e9275/687474703a2f2f692e7974696d672e636f6d2f76692f3051493378675875422d512f687164656661756c742e6a7067" height="220px" align="left">
</p>

## TensorFlow
TensorFlow™ is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. TensorFlow was originally developed by researchers and engineers working on the Google Brain Team within Google's Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research, but the system is general enough to be applicable in a wide variety of other domains as well.



Dependencies
============

TensorFlow (https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#pip-installation)

Use [pip](https://pypi.python.org/pypi/pip) to install any missing dependencies

Credits
===========
Credit for the vast majority of code here goes to Google! I've merely created a wrapper around all of the important functions to get people started.