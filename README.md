# Machine Learning hw1 <span style="color:red"></span>
Author: 洪彗庭 </br>
Student ID: A061610 

## Overview
In this homework, we are going to deal with 3 mathematical problems and 2 programming problems. The former 3 problems is finished in hand-written format, and this repository mainly show the result of the latter 2 programming problems' result. 

##### Files: </br>
* [hw1-1.pdf](./hw1_1.pdf): for solving Jensen's Inequality.
* [hw1-2.pdf](./hw1_2.pdf): for solving Entropy of the Univariate Gaussian.
* [hw1-3.pdf](./hw1_3.pdf): for solving KL divergence between two Gaussians.
* [hw1-4-polynomial-curve-fitting.ipynb](./hw1-4_polynomial_curve_fitting.ipynb): for solving Polynomial Curve Fitting problem. </br>
* [hw1-5-polynomial-regression.ipynb](./hw1-5_polynomial_regression.ipynb): for solving Polynomial Regression problem. </br>
* [environment.yml](./environment.yml): build conda environment to avoid tedious dependencies problem.

## Setup

* python 3.6.2
* scipy 0.19.1 (for loading the .mat files)
* numpy 1.13.1
* ipython

### Other Packages
* pandas

To avoid tedious dependencies problem, I will recommand you to use [Anaconda](https://anaconda.org/) to create a small custom environment as follows:

```
conda env create -f environment.yml
# activate the environment
source activate ML_hw1
# deactivate when you want to leave the environment
source deactivate ML_hw1
```

## Results and Discussions
### Problem 4 : Polynomial Curve Fitting
* Root-Mean-Square error evaluated on traning set and testing set. (Order, M,  range from 1 to 9)
![alt text](./OutputFigure/hw1-4-(1)_non_regularize.png)

As the figure shows, We believe when order goes to 7 or more will cause the overfitting problem. (more detail discussion please refer to the [report](./report.pdf)).
 
* Training and testing Root-Mean-Square error in regularized case. (Order, M,  fix to 9; Regularized term, lnλ, range from -20 to 0)
![alt text](./OutputFigure/hw1-4-(2)_regularize.png)

More Detail discussion of the effect of λ please also refer to the [report](./report.pdf).

### Problem 5 : Polynomial Regression
* Root-Mean-Square error evaluated on **multi-dimension** traning set and testing set. (Order, M, fix to 1 and 2)

	![alt text](./OutputFigure/hw1-5-(1)_non_regularize_multi-dim.png)

As the figure shows, when order increases, the RMS error will decrease.
 
* Training Root-Mean-Square error of different attribute. (Selected dimension range from 0th to 3th)

	![alt text](./OutputFigure/hw1-5-(2)_atribute_decision.png)

As the figure shows, we will select the 3rd dimension as the most contributive attribute, since it achieve the lowest RMS error on the traning set. More detail please also refer to the [report](./report.pdf).




## Implementation
### Data 
The data provided in this assigment are as follows:

* problem-4 : **1-dimension** data


|| target(label) | data|
|-------|----------|-------|
| training(# 20)| 1.976161</br> -1.016098</br> ...  | 6.283185 </br> 4.983216 </br> ...|
| testing(# 10)| -4.433719</br> -0.695587</br> ...  | 2.383277 </br> 5.199877 </br> ... |

* problem-5 : **4-dimension** data </br>
(3-classes, # of 50 data per class)</br>
(traning: first 80% of data in each class)</br>
(testing: last 20% of data in each class)


|| target(label) | data|
|-------|----------|-------|
| training(# 120)|1 </br> 1</br> ...</br>2</br> 2</br> ...</br>3</br> 3</br> ...   | [ 5.1  3.5  1.4  0.2] </br> [ 4.9  3.   1.4  0.2] </br> ...</br>[ 7.   3.2  4.7  1.4]</br>[ 6.4  3.2  4.5  1.5]</br>...</br>[ 6.3  3.3  6.   2.5]</br>[ 5.8  2.7  5.1  1.9]</br> ...|
| testing(# 30)| 1</br> 1</br> ...</br>2</br> 2</br> ...</br>3</br> 3</br> ...  | [ 5.   3.5  1.3  0.3]</br>[ 4.5  2.3  1.3  0.3] </br> ...</br>[ 5.5  2.6  4.4  1.2]</br>[ 6.1  3.   4.6  1.4]</br>...</br>[ 6.7  3.1  5.6  2.4]</br>[ 6.9  3.1  5.1  2.3]</br> ... |



### Solving the Weight
recalll that in this assignment we define the polynomial as :

$ y(x, w) = w_{0} + w_{1}*x + w_{1}*x^{2} + ... + w_{M}*x^{M} = \sum_{j=0}^{M}w_j*x^{j} $

### Training Detail
1. **AlexNet**:</br>
I didn't fintune all layers, since sometimes it will lose the advantage of loading pretrain parameters into the netwrok. I **freeze the first 2 bottom layers** (i.e. conv1, conv2) and finetune from conv3 to fc6 and also finetune on the additional 2 layers I add above the concate result.
2. **InceptionV3**:</br>
**First**: finetune the 2 layers I add above the whole structure (i.e. **freeze all layers in the InceptionV3**) </br>
**Second**: finetune the 2 layers I add on top and 2-stages of InceptionV3(i.e. **freeze the bottom 249 layers**)</br>
In this way, we can first avoid that since the layers we initialize is too bad (think of it as random generates), it prones to ruin the parameters if we directly finetune them with loaded weight InceptionV3 model. Also, on the second time of finetuning, it can converge more easily since we have already trained the first top 2 layers which are initially pretty worse.


## Discussion
1. Preprocess of data (shuffle do important ! )</br>
Initially I divide the train/val data in a wrong way, which I didn't apply shuffle on data before divide into train-set and validation-set. The result between non-shuffle and shuffle data is as follows: </br>

| Best-val-loss / Best-val-Acc | non-shuffle | shuffle |
|-------|----------|----------|
| Model-InceptionV3| 1.7323 / 0.6119 | 0.1381 / 0.9579 |

Especially the data we get this time is the sequence frames of the vedio, so the drawback of un-shuffle data will be more obvious in this task.

2. One stream v.s. Two streams 

| Accuracy | 1-stream | 2-streams |
|-------|----------|----------|
| Model-AlexNet| 0.4175 | 0.5658 |
One thing need to notice is that I can't confirm that the 2-streams result will definitely beat the result of 1-stream, since the setting of 1-stream and 2-streams are as follows: </br>

| | learning-rate | finetune-layers | epoch | batch-size |
|-------|----------|----------|--------|------|
| 1-stream|  0.001 | fc7, fc8 | 10 | 128 |
| 2-streams|  0.001 | conv3, conv4, conv5, fc6, fc7, fc8 | 10 | 16 |

The setting on the two is a little bit different (finetune-layers and batch-size), so I am not 100 percent for sure that 2-streams is better than 2-stream.

## Acknowledgement
Thanks the awesome tutorial of finetuning on AlexNet done by [Frederik Kratzert](https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html). </br>
Also thanks [Zayn Liu](https://github.com/refu0523) giving me so many advice on finishing this assignment.

## Reference
[scikit-learn lib on drawing precision-recall-curve](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

[code of little data for fintuning](https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975)

[keras example](https://keras.io/applications/#resnet50)

[alexnet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

[Inceptionv3](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)
