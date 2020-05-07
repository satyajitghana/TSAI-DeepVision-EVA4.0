# TinyImageNet and Annotations Assignment


1.  Assignment A:
    1.  Download this  [TINY IMAGENET (Links to an external site.)](http://cs231n.stanford.edu/tiny-imagenet-200.zip)  dataset.
    2.  Train ResNet18 on this dataset (70/30 split) for 50 Epochs. Target 50%+ Validation Accuracy.
    3.  Submit Results. Of course, you are using your own package for everything. You can look at  [this (Links to an external site.)](https://github.com/sonugiri1043/Train_ResNet_On_Tiny_ImageNet/blob/master/Train_ResNet_On_Tiny_ImageNet.ipynb)  for reference.
2.  Assignment B:
    1.  Download 50 images of dogs.
    2.  Use  [this (Links to an external site.)](http://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html)  to annotate bounding boxes around the dogs.
    3.  Download JSON file.
    4.  Describe the contents of this JSON file in FULL details (you don't need to describe all 10 instances, anyone would work).
    5.  Refer to this  [tutorial (Links to an external site.)](https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203). Find out the best total numbers of clusters. Upload link to your Colab File uploaded to GitHub.

Questions in S12-Assignment-Solution:

1.  What is your final accuracy?
2.  Share the Github link to your ResNet-Tiny-ImageNet code. All the logs must be visible.
    
3.  Describe the contents of the JSON file in detail. You need to explain each element in detail.
4.  Share the link to your Github file where you have calculated the best K clusters for your 50 dog dataset.
    
5.  Share the link to your 50 Dog Images Folder on GitHub
6.  Share the link to your JSON file on GitHub

# Solution

## TinyImageNet Model

Github Link: [https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/12_TinyImageNet/TinyImageNet.ipynb](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/12_TinyImageNet/TinyImageNet.ipynb)

Colab Link: [https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/12_TinyImageNet/TinyImageNet.ipynb#scrollTo=CDyM79bXAaMl](https://colab.research.google.com/github/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/12_TinyImageNet/TinyImageNet.ipynb#scrollTo=CDyM79bXAaMl)
```
Final Accuracy
Test: 53.66
Train: 63.22
```

## Doggo Dataset

Images: [https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/tree/master/12_TinyImageNet/Doggos/images](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/tree/master/12_TinyImageNet/Doggos/images)


(includes the explanation of json file)
KMeans run on Doggo: [https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/12_TinyImageNet/Doggos/Doggo_KMeans.ipynb](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/12_TinyImageNet/Doggos/Doggo_KMeans.ipynb)

Donno Annotations: [https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/12_TinyImageNet/Doggos/annotations.json](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/12_TinyImageNet/Doggos/annotations.json)

## Visualization

**Annotation**

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/12_TinyImageNet/Doggos/annotated.png?raw=true)


**KMeans**

![enter image description here](https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/blob/master/12_TinyImageNet/Doggos/kmeans.png?raw=true)


