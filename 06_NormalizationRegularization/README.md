# Normalization & Regularization

Code for the Experiments

`google-colab link`: [06_NormalizationRegularization_Modularized_FINAL](https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/06_NormalizationRegularization/06_NormalizationRegularization_Modularized_FINAL.ipynb)

`github link`: [06_NormalizationRegularization_Modularized_FINAL.ipynb](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/06_NormalizationRegularization/06_NormalizationRegularization_Modularized_FINAL.ipynb)

>Dataset: MNIST Digits
Model: CNN - 9960 params
Epochs: 40

Regularization Values
```
l1_lambda = 5e-4
l2_lambda = 5e-3
```

# Results

## 1. No L1 No L2

> `Highest Accuracy: 99.47`

### Metrics
![nol1_nol2_metrics.png](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/06_NormalizationRegularization/plots/nol1_nol2_metrics.png?raw=true)

### Misclassifications
![nol1_nol2_misclassifications.png](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/06_NormalizationRegularization/plots/nol1_nol2_misclassifications.png?raw=true)

## 2. L1

> `Highest Accuracy: 99.44`

### Metrics
![l1_metrics.png](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/06_NormalizationRegularization/plots/l1_metrics.png?raw=true)

### Misclassifications
![l1_misclassifications.png](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/06_NormalizationRegularization/plots/l1_misclassifications.png?raw=true)

## 3. L2

> `Highest Accuracy: 99.52`

### Metrics
![l2_metrics.png](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/06_NormalizationRegularization/plots/l2_metrics.png?raw=true)

### Misclassifications
![l2_misclassifications.png](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/06_NormalizationRegularization/plots/l2_misclassifications.png?raw=true)

## 4. L1 & L2

> `Highest Accuracy: 98.76`

### Metrics
![l1_l2_metrics.png](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/06_NormalizationRegularization/plots/l1_l2_metrics.png?raw=true)

### Misclassifications
![l1_l2_misclassifications.png](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/06_NormalizationRegularization/plots/l1_l2_misclassifications.png?raw=true)

# Comparison

## Validation Accuracy

![val_acc_compare2.png](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/06_NormalizationRegularization/plots/val_acc_compare2.png?raw=true)

![val_acc_compare.png](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/06_NormalizationRegularization/plots/val_acc_compare.png?raw=true)

## Validation Loss

![val_loss_compare2.png](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/06_NormalizationRegularization/plots/val_loss_compare2.png?raw=true)

![val_loss_compare.png](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/06_NormalizationRegularization/plots/val_loss_compare.png?raw=true)

