# Sketch-Classification

## Introduction
Deep networks has achieved high performance on general image recognition tasks by extracting the details in the picture. However, the high performance is hard to transfer to many cases (like cartoon, sketches) where the pictures are missing many visual details. In this project, we mainly focused on the free-hand sketches classification task. The dataset we used is a subset of large-scale sketch QuickDraw set of 25 categories. On processing the data, we have tried different models including Fully Convolutional Network, CNN, AlexNet, ResNet, DenseNet and RNN as well as compared the performance of them. RNN performs best (77.58\%) among all these models. Besides, some tricks like data augmentation and SIFT feature extraction have been used on RNN model.

## Usage
The models could be checked at folder /code/.

SIFT method could be checked at folder /data/.
