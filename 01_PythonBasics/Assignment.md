
# Python101 Assignment

1.  What are Channels and Kernels (according to EVA)?
2.  Why should we (nearly) always use 3x3 kernels?
3.  How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)
4.  How are kernels initialized?
5.  What happens during the training of a DNN?

# Answers

Link to the Colab-Notebook : [Python101.ipynb](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/01_PythonBasics/Python101.ipynb)

## 1. Channels and Kernels

To elaborate on channels, consider an example of text detection, like an OCR, let's say we have a huge corpus of text, consisting of letters from the English alphabet (A-Z), to recognize these letters i'll have 26 channels, each channel will be responsible to capture information of a specific letter. This is my channel, a channel is grouping/container of similar characteristics/information, in the example one of my channel will be an ``e`` channel which will capture the e's from my image, the channel will have all sorts of e's in it, of varying sizes and skew-ness. Here ``e`` is my feature, the channel associated with e 

Note: In the above explanation we are taking about a basic-intermediate channel, we could meaningful letters to form another channel, which would be a complex channel. But the explanation still serves the purpose.
RGB are also channels, because R channel has all the red-ness of the image, G channel has all the green-ness of the image. But we generally won't work with colours, because they don't provide us with much information. Consider the below example,

![Optical Lines Illusion](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/01_PythonBasics/015-colour-grid-optical-illusion-2.jpg?raw=true)

The actual image is colour-less and has only lines of colours, our brain fills in the remaining colours. also let's say we want to recognize a banana from it's image, here "yellow" is of less importance, the texture and shape of the banana is important, since the banana can be of any color, yellow-ness cannot be the major factor to determine if it's a banana.

A Kernel/Filter is an extractor of feature, for example in the e channel, my kernel will somewhat look like an e. Every kernel is associated with a channel, so if i apply the "e" kernel/filter to my image, then the output is going to be my "e-channel".

More on Sobel Operators: The operator uses two 3×3 kernels which are convolved with the original image to calculate approximations of the derivatives – one for horizontal changes, and one for vertical. If we define **A** as the source image, and **G**_x_ and **G**_y_ are two images which at each point contain the vertical and horizontal derivative approximations respectively

Here's an example of Sobel-X, Sobel-Y and Laplacian Filter that extract the vertical, horizontal lines and edges respectively.

![gradients](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/01_PythonBasics/gradients.jpg?raw=true)

These are also called image gradient, and the following kernels are approximation of those gradients,
$$ Laplacian = \nabla^2 f = \frac{\partial ^2{f}}{\partial x^2} + \frac{\partial ^2{f}}{\partial y^2} = \left[\begin{array}{ccc} 0 & 1 & 0\\ 1 & -4 & 1\\ 0 & 1 & 0 \end{array}\right] $$

$$ Gradient = \nabla f = \left[\begin{array}{c} g_x \\ g_y\end{array} \right]$$
$$ G_x = \left[\begin{array}{ccc} -1 & 0 & 1\\ -2 & 0 & 2\\ -1 & 0 & 1 \end{array}\right]$$
$$ G_y = \left[\begin{array}{ccc} 1 & 2 & 1\\0 & 0 & 0\\ -1 & -2 & -1 \end{array}\right]$$


## 2. 3x3 Kernels

It can be explained by why we do not use a ``2x2`` kernel, a kernel is applied for over a group of neighboring pixels and the kernel makes use of the information provided by those neighboring pixels to output a meaningful value/pixel, but in a ``2x2`` kernel we cannot make use of symmetric structure of the neighboring pixels, i.e. either we will make use of the neighbors to the right of our output pixel or the left, but if we take a ``3x3`` kernel we can make use of a symmetric neighborhood, i.e. the output pixel is surrounded by 8 pixels, 1 on each direction. For every ``odd x odd``  convolution kernel we can capture that symmetric information, but not so in an ``even x even`` kernel, so odd kernels are usually preferred.

Another explanation for using ``3x3`` kernel rather than any $(2n+1) \times (2n+1)$, i.e. any odd kernel, is because it helps in reducing the number of computations with the same receptive field.

The ​Receptive Field or Region of influence​ of the neuron/pixel in a subsequent layer refers to all  
the neurons/pixels from the previous that were involved in the convolution operation to produce  
said output neuron/pixel.

Consider an example, about receptive fields, lets say we want a ``1x1`` output pixel from a ``5x5`` input, i.e. the output pixel should have a receptive field of ``5x5``, one way to achieve this would be to use a ``5x5`` kernel, with ``25`` parameters, but another way would be to use 2 ``3x3`` kernels stacked together, hence it'll be ``5x5 > 3x3 > 1x1`` where each ``>`` denotes a convolution by a ``3x3`` kernel. The number of parameters will be ``2x9 = 18``  which is a reduction in number of parameters i need to train, reducing the computational cost while still preserving the receptive field of the output pixel, i.e. the output pixel will have all the information about the input ``5x5`` image.

![convolution](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/01_PythonBasics/convolution.png?raw=true)

Generally speaking we use a ``5x5`` or a ``7x7`` kernel for the first layer, since there are only 3 channels, but after this layer we tend to use stacks of ``3x3`` kernels.


## 3.  We perform a ``3x3`` convolution at every layer
```python
from __future__ import print_function
conv_layers = ['{}x{} >'.format(l_size, l_size)  for l_size in  range(199,  0,  -2)]
print(*conv_layers)
print('\b\b\nNumber of 3x3 convolutions = {}'.format(len(range(199,  0,  -2))-1))
```
Output
```
199x199 > 197x197 > 195x195 > 193x193 > 191x191 > 189x189 > 187x187 > 185x185 > 183x183 > 181x181 > 179x179 > 177x177 > 175x175 > 173x173 > 171x171 > 169x169 > 167x167 > 165x165 > 163x163 > 161x161 > 159x159 > 157x157 > 155x155 > 153x153 > 151x151 > 149x149 > 147x147 > 145x145 > 143x143 > 141x141 > 139x139 > 137x137 > 135x135 > 133x133 > 131x131 > 129x129 > 127x127 > 125x125 > 123x123 > 121x121 > 119x119 > 117x117 > 115x115 > 113x113 > 111x111 > 109x109 > 107x107 > 105x105 > 103x103 > 101x101 > 99x99 > 97x97 > 95x95 > 93x93 > 91x91 > 89x89 > 87x87 > 85x85 > 83x83 > 81x81 > 79x79 > 77x77 > 75x75 > 73x73 > 71x71 > 69x69 > 67x67 > 65x65 > 63x63 > 61x61 > 59x59 > 57x57 > 55x55 > 53x53 > 51x51 > 49x49 > 47x47 > 45x45 > 43x43 > 41x41 > 39x39 > 37x37 > 35x35 > 33x33 > 31x31 > 29x29 > 27x27 > 25x25 > 23x23 > 21x21 > 19x19 > 17x17 > 15x15 > 13x13 > 11x11 > 9x9 > 7x7 > 5x5 > 3x3 > 1x1
Number of 3x3 convolutions = 99
```
Every ``>`` in the above output is a ``3x3`` convolution, there are total of 99 such ``>``.

## 4. Kernel Initialization

Every kernel is composed of weights, before going into how they are initialized, let's understand the problems associated with values of the weights,

1. A too-large initialization leads to exploding gradients
2. A too-small initialization leads to vanishing gradients

All in all, initializing weights with inappropriate values will lead to divergence or a slow-down in the training of your neural network. Although we illustrated the exploding/vanishing gradient problem with simple symmetrical weight matrices, the observation generalizes to any initialization values that are too small or too large. [1]

To prevent the vanishing and exploding gradients problem we will use the following rule of thumb,
1.  The mean of the activations should be zero.
2.  The variance of the activations should stay the same across every layer.

Under these two assumptions, the backpropagated gradient signal should not be multiplied by values too small or too large in any layer. It should travel to the input layer without exploding or vanishing.

Ensuring zero-mean and maintaining the value of the variance of the input of every layer guarantees no exploding/vanishing signal. This method applies both to the forward propagation (for activations) and backward propagation (for gradients of the cost with respect to activations).

As most values are concentrated towards the mean, most of the random values selected have higher probability to be closer to mean (say $\mu=0$). Also instead of drawing from standard normal distribution, we are drawing W from normal distribution with variance k/n, where k depends on the activation function. While these heuristics do not completely solve the exploding/vanishing gradients issue, they help mitigate it to a great extent.

Xavier and He Initialization are two commonly used method that deal with the above discussed problems and provide a pretty nifty solution based on the number of layers and the layer size that will result in a faster convergence than just picking any random value. Xavier initialization works with tanh activations, If you are using ReLU, for example, a common initialization is He initialization ([He et al., Delving Deep into Rectifiers](https://arxiv.org/pdf/1502.01852.pdf)), in which the weights are initialized by multiplying by 2 the variance of the Xavier initialization.

But why not initialize all the weights to equal values ? 
If we initialize the weights to equal values, then the equality carries through  to the gradients associated with these weights, thereby keeping the weights equal throughout  the training. This is known as the Symmetry Breaking Problem, where if you start with equal initialized weights, they remain equal through the training.

## 5. Training of a DNN

A deep learning model consists of various convolution layers, with pooling layers and finally are connected with a fully connected layer with a output layer. Just like any neural network we do a forward pass, calculate the gradients and then do a back-propagation to adjust our weights such that the predicted output is more close to the actual output, i.e. the loss decreases.

We'll concentrate more on the training part, there are various things that will be going on during the training.

Let's take the example of a CNN which is a type of DNN, in the network we have various convolutional layers that captures specific information about the input.

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/01_PythonBasics/channels-layers.png?raw=true)

In the above image we can see how these kernels work, the first layer is responsible for detecting the edges, and then detect the textures from those extracted edges, then patterns and then parts of an object and then the final object. The shapes that we see in the above image are the images that the channel is made of, it'll make the kernel most "happy" i.e. very low loss for that kernel.

Each of the kernel is composed of weights that are updated on every pass of an image, in such a way as to reduce the final loss. These updates are made during the back propagation along the gradient, backpropagation is a method to obtain the optimal value of a function by incremental methods. Various Optimization algorithms exists such as Stochastic Gradient Descent with Momentum, Nesterov Accelerated Gradient, AdaGrad, RMSProp and Adam Optimizer, which make sure that we reach the optimal value with the least epochs.

To reduce overfitting we introduce regularization, i.e penalizing the model, dropout is one such method that randomly dropping a few neurons.

## References:

1. Katanforoosh & Kunin, "Initializing neural networks", deeplearning.ai, 2019.
2. [https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)
3. [https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94](https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94)



<!--stackedit_data:
eyJoaXN0b3J5IjpbMTA1OTgzMzQ0LC04NzcwMTY0MzQsLTE5Nz
gwODY4MTQsLTExMzQ4MzE1OCwtMTc0NDI5OTE1NSwxOTgyMDY0
ODMyLC03NDkwNDAwNjUsNzg2MDkyNjgwLDk5NzMzNTU2Niw2ND
k2OTMxMjUsMTQzMDY0NDY2NSwtMjQ0Nzc3NzkwLDE5MTEwNTM1
ODMsNTc5MTA2ODgzLC0yMDI4ODE0NjM5LDE3MjcwOTg3NzAsLT
U5MDI0ODg0MywyMTE3MDA0MzMyLC0xODM1MTE5MDg5LDM2NjEy
NjgzNl19
-->