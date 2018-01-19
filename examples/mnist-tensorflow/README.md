# mnist-tensorflow

This sample project demonstrates basic kogu integration to Python using helper library.  
It trains and classifies MNIST hand-written digits, reporting resulting prediction accuracy.  
Its done using LeNet deep neural network model constructed with the help of TensorFlow.  
In addition to value logging, the script makes use of kogu file upload feature.  

## Instructions

* Make sure you have dependencies installed:  
  `pip install -r src/requirements.txt`  

* Run experiments (add [execution options](https://kogu.io/docs/cli.html) to taste):  
  `kogu run src/mnist-tensorflow.py`  

* Explore results in terminal or browser:  
  `kogu list`  
  `http://localhost:8193`  
