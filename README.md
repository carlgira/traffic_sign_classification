# Traffic Sign Recognition

---

This project is part of the Udacity self-driving car Nanodegree, for the construction of a neural network traffic sign classifier.

The project contains the following files:

- Readme.md (This file): Writeup of the project
- Traffic_Sign_Classifier.ipynb: Notebook with the python code and comments
- src: Directory with different architectures of neural networks (lenet with and without augmented data, lenet with dropout and other convnets)
- new_images: Folder with new images to test the model
- convent: Folder with checkpoint of the neural network
- signnames.csv: CSV file with labels id and descriptions.

*In the Traffic_Sign_Classifier.ipynb you can see all the Rubric points, in this document i'm only going to highlight the overal architecture, final results and conclusions*

### Data Set Summary & Exploration
The size and quality of the images is not the best, lots of them are pretty dark. (added a brigtness transformation in the data augmentation)

### Data augmenation
The distributions of the classes is not equal. There are some labels that has 2000 samples against others that only have 200.

Here it's important to augment the data using some image transformations (rotation, translation, image shear and brightness), and make all the classes with an equivalent number of samples.

The dataset after the image augmentation ends with a **5000 average samples by class**.

The data augmentation function applies randomly a range of image transformation to generate new samples

- Rotation
- Translation
- Image shear
- Brighness change

## Neural Network Models

I wanted to try to test some combinations of different neural networks to see how the changes improve the accuracy of the model.

All this networks where trained with the same parameters:

- Batch size, 256
- Learning rate 0.001
- EPOCHs, 100
- dropout rate, 0.75

| Neural Network         		|     Accuracy, Loss	        					| Test Accuracy   |
|:---------------------:|:---------------------------------------------:|:---------------------:| 
| Lenet no data augmentation         		| <img src="./images/lenet_no_aug.png" width=400 height=200 /> | 0.915 |
| Lenet with data augmentation         		| <img src="./images/lenet_data_aug.png" width=400 height=200 /> | 0.924 |
| Lenet, data augmentation, dropout         		| <img src="./images/lenet_dropout.png" width=400 height=200 /> | 0.939 |
| Lenet data augmentation, dropout, more filters in convolutions        		| <img src="./images/convnet.png" width=400 height=200 /> | 0.962 |

There are some information you can get from the final accuracy and the graphs

- The accuracy improve with the use of the data augmentation, dropout layer or using more filters in convolutions.
- The accuracy and loss when there is any data augmenation jumps a lot more that the others, combinations
- The dropout seems to help in the model making growth of the accuracy more smooth.
- I could use more EPOCHS to get better results 

#### Architecure

The last model (the one used in the notebook), has the next architecture. It's a basic lenet using more filters in the convolutions, dropout. 

- Convolution, 5x5, 16 filters, relu activation
- Max polling, 2x2.
- Convolution, 5x5, 32 filters, relu activation
- Max polling, 2x2.
- Full conected layer 120,  relu activation
- Dropout, 0.75 keep probability
- Full conected layer 84,  relu activation
- Dropout, 0.75 keep probability
- Full conected layer 43

- Loss function, softmax cross entropy
- Adam Optimier, learnin rate 0.001

Using this model i get 96% of accuracy on the test dataset


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?


### Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


