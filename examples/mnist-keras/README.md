# mninst-keras

This sample demonstrates basic kogu integration to Keras-powered Python script using helper library.  
It features a neural network that trains and classifies MNIST hand-written digits dataset.
Each epoch accuracy is logged to kogu with custom Keras callback.

## Instructions

* Make sure you have dependencies installed:  
  `pip install -r src/requirements.txt`  

* Run experiments (add [execution options](https://kogu.io/docs/cli.html) to taste):  
  `kogu run src/mnist-keras.py`  

* Explore results in terminal or browser:  
  `kogu list`  
  `http://localhost:8193`  
