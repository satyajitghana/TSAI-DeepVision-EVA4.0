# TSAI-DeepVision-EVA4.0
The contains the solutions to the assignments given in The School of AI - Extensive Vision AI 4.0

1. [Python Basics](01_PythonBasics/README.md)

This describes the very basics of Python, i would recommend binge watching Raymond Hettinger's YouTube Videos, as much as you can, also learn list comprehension, slicing, partial, functools, functional programming, classes, MRO, decorators, lambdas, python 3 typing

2. [Neural Architecture](02_NeuralArchitecture/README.md)

This describes the basic neural network architecure, see its page for details

3. [PyTorch 101](03_PyTorch101/README.md)

Basic Pytorch architecture for working with neural networks, introduces you to `nn.Module`, optimizers, forward and backward pass, datasets, how to apply simple augmentation.

4. [Architecture Basics](04_ArchitectureBasics/README.md)

Here we had to train MNIST to get 99.4% accuracy with some given contraints, was quite fun to do this, got me out of the noob shell of what a neural network actually is and what it does, because i had to write code manually, not copy paste any more.

5. [Coding Drill](05_CodingDrill/README.md)

This was very important to realise the basic steps required to make a neural network and then go on to optimize it, to get the perfect model size and accuracy, very important, please see its documentation, and all the notebooks

6. [Normalization Regularization](06_NormalizationRegularization/index.html)

This assigment focuses on the importance of normalization and regularization in neural networks, aim was to also get > 99.4% accuracy in less than 40 epochs for MNIST with limited model parameters

7. [Advanced Convolution](07_AdvancedConvolution/README.md)

Here i was introduced to various convolution types you can add in the network, we were given a custom network which we had to implement, also we had to make a custom library of python to support and ease the process of NN building and training, this was quite a eye openener for me, i made my first python package !, also i learnt the value of time contraints, since we have to finish the assignments within a week, along with my college work it was burdening, or maybe i was procrastinating?

8. [Advanced Architecture](08_AdvancedArchitecture/README.html)

Here i was introduced to ResNet18 and CIFAR dataset with aim to get 85% test accuracy

9. [Model Diagnostics](09_ModelDiagnostics/README.html)

This was amazing, i was introduced to GradCAM, and i myself learnt about saliency map, this was something i also used in my college group project, it's used to diagnosing if the model is actually learning anything from the dataset. Quite a nifty feature, eh ?

10. [Advanced Training](10_AdvancedTraining/README.html)

Here i had to use LR Finder to find the best learning rate for the model, i also used OneCyclePolicy for faster convergence. i reached 92.03% accuracy in CIFAR using ResNet18

11. [Super Convergence](11_SuperConvergence/index.html)

A custom architecture was implemented here, along with that i used OneCycleLR to improve the convergence of the NN

12. [Tiny Image Net](12_TinyImageNet/index.html)

Here we had to do image annotation and collect doggo dataset.

Also i had to make a custom dataloader and dataset class in PyTorch to support TinyImageNet, and then train a model to reach 60% accuracy, which i failed to, but still i learnt from my mistakes ! the one who doesnt make mistakes does really make anything does he ?

13. [YOLOV3](13_YoloV3/index.html)

I was introduced to YOLO, something i always wondered how it exactly worked, and why is it called YOLO ? now i understand why. and people complain about the FPS of YOLO, now i know why, they dont pay attention to the anchor boxes, which is very important and its different for different datasets.

I made a custom Bugs Bunny detector using the YOLO architecture :) i like my detector :) i like bugs bunny :) i custom collected bugs bunny images, self annotated about 600 images of my bunny and trained the network for like 5 hours on a Tesla P100

14. [RCNN](14_RCNN/index.html)

I was introduced to RCNN family and SSD, but that wasnt the main thing here.

The main thing was that i created a dataset containing 1.2M images ! do you understand how crazy that is ? this was a pre-req for the upcoming CapStone project which will be to do object segmentation and also monocular depth estimation both at the same time.

15. [CapStone Project](15_CapstoneProject/README.md)

This was the final project, please see its README, i spent a lot of time writing code and documenting it.

I'm sure you'll be impressed by this one ;)


---
<h3 align="center">Made with 💘 by shadowleaf</h3>
