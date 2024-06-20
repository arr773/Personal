# Image transformations and deep learning

# Commands to run the code.

## Best Performing Model for CIFAR10
To run the best model we prepared for the Plain Old CIFAR10 data
```python
python3 main.py --epochs 50 --model_class 'Plain-Old-CIFAR10' --batch_size 128 --learning_rate 0.001 --l2_regularization 0.0
```

## D-shuffletruffle Model
To run the 16 patch D-shuffletruffle model
```python
python3 main.py --epochs 100 --model_class 'D-shuffletruffle' --batch_size 128 --learning_rate 0.1 --l2_regularization 0.0
```

## N-shuffletruffle Model
To run the 8 patch N-shuffletruffle model
```python
python3 main.py --epochs 100 --model_class 'N-shuffletruffle' --batch_size 128 --learning_rate 0.001 --l2_regularization 0.001
```
# Approach to the problem
We started approaching the problem using established convolutional neural network (CNN) architectures available on GitHub, including AlexNet, DenseNet, ResNet, VGGNet, and GoogleNet, evaluated for their performance on the CIFAR-10 dataset using patch sizes of 16x16 and 8x8. However, the initial results revealed that these models didn't perform well, mainly due to the overlapping of kernels within the convolutional layers. This issue of overlapping kernels, occurring when adjacent kernels share common pixels during the sliding window operation, led to extraction of redundant features consequently decreasing the  accuracy, especially in 16 patch and 8 patch. Also the because of the overlapping patches features being extracted from the images. To fix this, we modified architecture to be invarient of the shuffled patches for both D-shuffletruffle and N-shuffletruffle. After making these changes, we re-evaluated the models and saw a significant improvement in accuracy across all patch sizes and was similar for all the three test set.

## Normal Images
We used the architecture of googlnet to get good accuracy, it has 9 inception layers along with maxpool and average pool layer. We then used single layer of feed forward network after getting feature set from convolutional layers. This gave us the accuracy of 87%.

## D-shuffletruffle Model
For D-shuffletruffle model using normal cnn architecture was not sufficient. For this we had to have seperate features extracted from each patch individually and then take the average of the feature vectors before sending them to feed forward layer. We used 3 layers of convolutional network with kernel size 5 along with dropout and maxpooling layer, and then the merged features were forwarded to feed forward layer of 5 layers. This helped us to get same accuracy for normal images and D-shuffletruffle images. 

## N-Shuffletruffle Model
For N-Shuffletruffle model we used the same approach that we used for 16 patches but the accuracy number didn't exceed 52% even with different number of layers or kernel size. Then we moved on to Vision Transformer model. Here we removed the positional encoding to remove the order of the patches of 8x8. By this we got same accuracy number for normal images and shuffled ones. This is because of removing positional encoding the same weights are shared between the patches. This gave us a good accuracy ogf 58%.


# Results for 3 best working models

| Accuracy  | Normal | D-shuffletruffle | N-shuffletruffle |
| :---:   | :---: | :---: |  :---: |
| GoogleNet | 87.020   |  57.5  |  27.88   |
| Custom CNN  | 72.680   | 72.68   |  36.41   |
| VIT     |   55.070   |   55.07  |  55.07   |

# Graphs for some of the models tested and tried
(We tried a lot of models but didnt record the graphs for many of them so we will be discussing them in a later section)
The first graph in each in the training loss while the second graph is the validation loss
## GoogleNet
![image](https://media.github.iu.edu/user/25599/files/e02e1ac6-a4dc-4179-a2fd-4a695441cd2b)
## Custom CNN for 16x16 patch
![image](https://media.github.iu.edu/user/25599/files/ba7fa523-77a7-4b3e-a3d6-58ea527b2e95)
## VIT for 8x8 patch
![WhatsApp Image 2024-04-12 at 22 06 56_d3e3f1b9](https://media.github.iu.edu/user/25599/files/78b1e78c-9742-4a89-b0b8-4581ef3c0407)
## CVT
![image](https://media.github.iu.edu/user/25599/files/79af01b0-6c02-4eb6-8293-e5df3742a347)
## Custom CNN for 8x8 patch
![image](https://media.github.iu.edu/user/25599/files/8f7b9bfc-88b2-49e3-86af-de6e5e61e78b)

# PCA

## GoogleNet
![WhatsApp Image 2024-04-12 at 22 52 39_da306603](https://media.github.iu.edu/user/25599/files/dd1c6287-a028-44b7-af78-f6f572aef349)

## 16 Patch
![WhatsApp Image 2024-04-12 at 22 49 56_426239a1](https://media.github.iu.edu/user/25599/files/af6aa84c-04f9-4617-8fff-1d4486366ea8)

## 8 Patch VIT
![WhatsApp Image 2024-04-12 at 22 51 02_eaff9ef9](https://media.github.iu.edu/user/25599/files/6acadd3c-b0bc-4af5-9892-a808f5bec8f2)

## Analysis
As we can observe from the above graphs the data points in the GoogleNet are scattered throughout. This is because of the distinct features between the original, 16 patch and 8 patch data points. The 16 patch custom CNN has data points scattered but not as much as that of the GoogleNet. This is because of the fact that the 16 patch and the original data points and similar fearues. The VIT graph has a lot of overlapping data points because of the similar features in the original data, the 16 patch data and the 8 patch data. We have created a few custom CNN models which try to prevent the overlapping between kernels so as to test the 8 patch and 16 patch data which we believe will give similar accuracies.

# All models tried and individual contributions 

### Aaryan Agarwal
Tried the LeNet model as a starter but due to a low test accuracy of near 30% the idea was dropped. Researched the some of the best architectures that can be used for CIFAR10 data. After researching and trying a few models, he came across GoogleNet. Initially GoogleNet was giving a test accuracy of around 80% but after tweaking the architecture and customizing the architecture based upon the inception model, the accuracy finally crossed 87%. Also reseached a few other models for the 8 patch and 16 patch datasets but their accuracy wasn't crossing 25% and 50% respectively. Worked on the 75 data point creation and PCA analysis in the latter part of the assignment. 
### Kishan Singh Rathore
Started with Swin Transformer and gained good accuracy of 62% as a starting point. After that worked with CVT and increaesd the normal test accuracy to 75.15 with 16 and 8 as 57 and 38. But that wasnt enough for him so he moved forward with VIT and created the model for the 8 patch with an accuracy of 55%. Also created the custom CNN for the 16 patch which was crossing 72%. Worked on the PCA and visualization of the PCA in the form of graphs.
### Sumukha Sharma Thoppanahalli Chandramouli
AlexNet was the first algorithm he came across with an accuracy around 70% for the original testset. Created a few custom models for the 16 patch and 8 patch where the accuracies were nearing 61% and 55% respectively. He achieved this by trying to prevent the kernels from overlapping each other while training the data. Tried ResNet but the accuracy was lesser than that of GoogleNet so we discarded it. The accruacy of ResNet was somewhere around 80% so as to understand better why it was dropped. Also worked for coming up ideas for the PCA and the 75 data points testset. 

# References
https://github.com/soapisnotfat/pytorch-cifar10/tree/master 
https://github.com/mashaan14/YouTube-channel/blob/main/notebooks/2024_01_08_CNN_and_ViT.ipynb

