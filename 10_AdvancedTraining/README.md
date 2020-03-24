# Advanced Training

Assignment:

1.  Pick your last code
2.  Make sure to Add CutOut to your code. It should come from your transformations (albumentations)
3.  Use this repo:  [https://github.com/davidtvs/pytorch-lr-finder (Links to an external site.)](https://github.com/davidtvs/pytorch-lr-finder)
    1.  Move LR Finder code to your modules
    2.  Implement LR Finder (for SGD, not for ADAM)
    3.  Implement ReduceLROnPlatea:  [https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau (Links to an external site.)](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
4.  Find best LR to train your model
5.  Use SDG with Momentum
6.  Train for 50 Epochs.
7.  Show Training and Test Accuracy curves
8.  Target 88% Accuracy.
9.  Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.
10.  Submit

# Solution

Github-Notebook : [https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/10_AdvancedTraining/AdvancedTraining.ipynb](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/10_AdvancedTraining/AdvancedTraining.ipynb)

Google-Colab: https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/10_AdvancedTraining/AdvancedTraining.ipynb

## Model Stats

```
Epochs: 50
Max Train Accuracy: 98.27
Max Test Accuracy: 92.03
```


## Model Metrics

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/10_AdvancedTraining/metrics_1.png?raw=true)

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/10_AdvancedTraining/metrics_2.png?raw=true)

## Grad-CAM

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/10_AdvancedTraining/gradcam.png?raw=true)

## Misclassifications Grad-CAM

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/10_AdvancedTraining/misclassified_gradcam.png?raw=true)

