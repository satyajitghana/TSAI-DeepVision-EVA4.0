


# Assignment:

1.  Go through this repository:  [https://github.com/kuangliu/pytorch-cifar (Links to an external site.)](https://github.com/kuangliu/pytorch-cifar)
2.  Extract the ResNet18 model from this repository and add it to your API/repo.
3.  Use your data loader, model loading, train, and test code to train ResNet18 on Cifar10
4.  Your Target is 85% accuracy. No limit on the number of epochs. Use default ResNet18 code (so params are fixed).
5.  Once done finish S8-Assignment-Solution.

# Solution
Github-Notebook : https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/08_AdvancedArchitecture/ResNet18.ipynb

Colab-Notebook : https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/08_AdvancedArchitecture/ResNet18.ipynb

PySodium Library : [https://github.com/satyajitghana/PySodium](https://github.com/satyajitghana/PySodium)
```
Model : ResNet18
Data Augmentation : RandomCrop, RandomHorizontalFlip
Epochs : 15
Scheduler : OneCycleLR
Optimizer : SGD
Train Accuracy (Best) : 95.63
Test Accuracy (Best) : 90.84
```

## Conclusions : 
- It was quite fun to create a library, there are a lot of design choices you have to make and stick with it throughout. You get to learn how to write generic functions and how to export them.
- ResNet18 model overfits, so we added data augmentation to fix that
- OneCycleLR helps achieve high accuracy within very less epochs
- The model still overfits, maybe add L2 Regularization ?

