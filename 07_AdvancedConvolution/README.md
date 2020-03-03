# Advanced Convolution

1. change the code such that it uses GPU
2. change the architecture to C1C2C3C40 (basically 3 MPs)
3. total RF must be more than 44
4. one of the layers must use Depthwise Separable Convolution
5. one of the layers must use Dilated Convolution
6. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
7. achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M. 

> Link to notebook : https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/07_AdvancedConvolution/AdvancedConvolution.ipynb \

> Link to google-colab   : https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/07_AdvancedConvolution/AdvancedConvolution.ipynb

> PySodium Library : https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/tree/master/07_AdvancedConvolution/PySodium

# Lessons Learnt

1. I need to practice more on network architecture, i have no clue why the network works, probably because backrop saves me, and i just added a huge number of parameters, the network was forced to learn.

2. Kudos to Backprop for saving me

3. Writing Python libraries takes time, invest more time in writing good and intuitive library

4. Write good documentation for the library
