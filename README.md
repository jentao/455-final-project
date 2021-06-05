# 455-final-project: Bird Classification Challenge!
Contributor: Amber Xu,
             Jennifer Tao,
             Kenny Wu,
             Zuer Wang
             
## Overview
Since we are not good at distinguishing hundreds of different types of birds, we plan to train a computer vision model to help us recognize the birds. Hence we worked on the Kaggle Bird Classification Challenge. 

The purpose of this model is to classify images of birds based on the training set and their labels. There are 555 different image categories (birds) given by integers in the dataset, and we need to predict the labels of images in the training set. We used various ResNet pre-trained model architectures and experimented with augmenting training data to decrease overfitting. The challenge was to write a classifier, that given an unidentified picture of the bird never before seen it could predict the class of that bird with high accuracy. 
The goal of this problem was to reach ~80% accuracy or higher. To improve the accuracy of the identification, we trained three models using Resnet architecture (Resnet18, Resnet34, Resnet50).

## Problem setup & Data used
For this competition, the data that we used to train the models is provided by the instructor, Joseph Redmon.
A [starter code](https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav#scrollTo=yRzPDiVzsyGz) also provided by our instructor, including loading the data, training, making predictions, visualizing loss images.

## ResNet18

load a ResNet-18 model pre-trained on the image, train the model for five epochs on the training data, and generate predictions for the test images. 

## ResNet34

## ResNet50

