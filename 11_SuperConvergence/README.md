# Super Convergence

## Assignment

1.  Write a code that draws this curve (without the arrows). In submission, you'll upload your drawn curve and code for that
    1.  ![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/11_SuperConvergence/assets/11s11.png?raw=true)
2.  Write a code which
    1.  uses this new ResNet Architecture for Cifar10:
        1.  PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        2.  Layer1 -
            1.  X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
            2.  R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
            3.  Add(X, R1)
        3.  Layer 2 -
            1.  Conv 3x3 [256k]
            2.  MaxPooling2D
            3.  BN
            4.  ReLU
        4.  Layer 3 -
            1.  X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
            2.  R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
            3.  Add(X, R2)
        5.  MaxPooling with Kernel Size 4
        6.  FC Layer
        7.  SoftMax
    2.  Uses One Cycle Policy such that:
        1.  Total Epochs = 24
        2.  Max at Epoch = 5
        3.  LRMIN = FIND
        4.  LRMAX = FIND
        5.  NO Annihilation
    3.  Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
    4.  Batch size = 512
    5.  Target Accuracy: 90%.
    6.  The lesser the modular your code is (i.e. more the code you have written in your Colab file), less marks you'd get.
3.  Questions asked are:
    1.  Upload the code you used to draw your ZIGZAG or CYCLIC TRIANGLE plot.
    2.  Upload your triangle Plot which was drawn with your code.
    3.  Upload the link to your GitHub copy of Colab Code.
    4.  Upload the github link for the model as described in A11.
    5.  What is your test accuracy?

## Solution

Github link: [https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/11_SuperConvergence/SuperConvergence.ipynb](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/11_SuperConvergence/SuperConvergence.ipynb)

Colab link: https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/11_SuperConvergence/SuperConvergence.ipynb

PySodium: [https://github.com/satyajitghana/PySodium](https://github.com/satyajitghana/PySodium)

Triangle Pattern: [https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/11_SuperConvergence/CycleLR.ipynb](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/11_SuperConvergence/CycleLR.ipynb)

### code for pattern
![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/11_SuperConvergence/assets/pattern_code.PNG?raw=true)

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/11_SuperConvergence/assets/one_cycle_fig.png?raw=true)

## Model Stats

```
Test Accuracy: 89.97
Train Accuracy: 92.66
Params: 6,573,120
```

### LR Finder

```
[ 2020-04-11 17:52:58,135 - sodium.sodium.runner ] INFO: sorted lrs : [0.609391, 0.61039, 0.6083919999999999, 0.611389, 0.607393, 0.606394, 0.63037, 0.613387, 0.626374, 0.612388]
[ 2020-04-11 17:52:58,137 - sodium.sodium.runner ] INFO: found the best lr : 0.609391
```

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/11_SuperConvergence/assets/lr_finder.png?raw=true)


```
[ 2020-04-11 17:53:02,878 - sodium.sodium.runner ] INFO: using max_lr : 0.609391
[ 2020-04-11 17:53:02,880 - sodium.sodium.runner ] INFO: using min_lr : 0.02437564
[ 2020-04-11 17:53:02,880 - sodium.sodium.runner ] INFO: using initial_lr : 0.02437564
```
### Learning Rate

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/11_SuperConvergence/assets/lr_metric.png?raw=true)

### Model Accuracy-Loss Curves

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/11_SuperConvergence/assets/model_stats.png?raw=true)
