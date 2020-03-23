# Model Diagnostics

## Quiz DNN

Github-Notebook : [https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/09_ModelDiagnostics/QuizDNN.ipynb](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/09_ModelDiagnostics/QuizDNN.ipynb)

Google-Colab : https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/09_ModelDiagnostics/QuizDNN.ipynb

## Assignment


1. Move your last code's transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
2. Please make sure that your test_transforms are simple and only using ToTensor and Normalize
3. Implement GradCam function as a module. 
4. Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
5. Target Accuracy is 87%
6. Submit answers to S9-Assignment-Solution.


Github-Notebook link : [https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/09_ModelDiagnostics/ModelDiagnostics.ipynb](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/09_ModelDiagnostics/ModelDiagnostics.ipynb)


Google-Colab link : https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/09_ModelDiagnostics/ModelDiagnostics.ipynb


Grad-CAM module : https://github.com/satyajitghana/PySodium/tree/v0.0.2/sodium/gradcam 


Albumentations code : https://github.com/satyajitghana/PySodium/blob/v0.0.2/sodium/data_loader/augmentation.py


## Model Stats

```
Epochs: 20
Max Train Accuracy: 97.01
Max Test Accuracy: 92.21
```


## Model Metrics

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/09_ModelDiagnostics/metrics.png?raw=true)

## Model Grad-CAM

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/09_ModelDiagnostics/gradcam_1.png?raw=true)

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/09_ModelDiagnostics/gradcam_2.png?raw=true)

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/09_ModelDiagnostics/gradcam_3.png?raw=true)
