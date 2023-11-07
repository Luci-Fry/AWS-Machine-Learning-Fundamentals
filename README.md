# AWS-Machine-Learning-Fundamentals
This repo contains the projects I worked on for the AWS Machine Learning Fundamentals Nanodegree on Udacity.

### Project 1: Prediction of bike sharing demand using Autogluon.
The models were able to predict the demand for bikes for a bike-sharing service based on factors such as season, weather, temperature, windspeed, holidays, and so on. 
Autogluon was used to construct many Machine Learning models including XGBoost, Random Forest, Logistic Regression, LightGBM, etc with various hyperparameters. 
The developed models were then ranked from best to worst to determine the best model for the use case. 


### Project 2: Prediction on MNIST handwritten digits dataset using a Deep Neural Network
A deep neural network was designed to evaluate the MNIST dataset. The network class utilizes ReLU activation and dropout to yield a **training loss of 0.186** after 13 epochs and a test accuracy of 
**97.47% test accuracy** after hyperparameter optimization.


### Project 3: Landmark classification using Convolutional Neural Networks (CNN)
First, a convolutional neural network was built from scratch for the classification of images of 50 popular landmarks around the world. 
The Residual Block characteristic of the Resnet architecture is implemented here in addition to Max pooling and Batch Normalization. 
Next, transfer learning is leveraged to construct a model built upon the resnet18 architecture in an attempt to improve model performance. This yielded a **test accuracy of 64 %** 

**NB:** The src directory contains all the rudimentary code (functions) for the model architectures, training, testing and other utilities while the jupyter notebooks contain high-level usage of these functions for a simpler flow. 


### Project 4: Image Classification Workflow on CIFAR-100 dataset with AWS
In this project, an image classification workflow is implemented using services on the Amazon Web Services (AWS) platform. These services include Amazon SageMaker for training, validating, and deploying the classification model to an endpoint, Lambda, 
and Step Functions for orchestrating and monitoring a serverless workflow using the deployed endpoint. Finally, inferences from the deployed model is analyzed. 
