import pickle
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import os.path
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

import numpy as np
import matplotlib.image as mpimg

import pandas as pd


signnames = pd.read_csv('signnames.csv')


import cv2

img_size = 32


def download_dataset():
	pass
	# https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
	return

def load_dataset(training_file, validation_file, testing_file):

	with open(training_file, mode='rb') as f:
		train = pickle.load(f)
	with open(validation_file, mode='rb') as f:
		valid = pickle.load(f)
	#with open(testing_file, mode='rb') as f:
	#	test = pickle.load(f)
	test = None

	return train, valid, test


download_dataset()
train, valid, test = load_dataset("traffic-signs-data/train.p", "traffic-signs-data/test.p", "traffic-signs-data/valid.p")

X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
#X_test, y_test = test['features'], test['labels']
X_test = None






#X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)
#X_validation_gray = np.sum(X_validation/3, axis=3, keepdims=True)
#X_test_gray = np.sum(X_test/3, axis=3, keepdims=True)


#X_train = X_train_gray
#X_validation = X_validation_gray
#X_test = X_test_gray


#X_test = (X_test - 128)/128



def transform_image(img,ang_range,shear_range,trans_range):
	'''
	This function transforms images to generate new images.
	The function takes in following arguments,
	1- Image
	2- ang_range: Range of angles for rotation
	3- shear_range: Range of values to apply affine transform to
	4- trans_range: Range of values to apply translations over.

	A Random uniform distribution is used to generate different parameters for transformation

	'''
	# Rotation

	ang_rot = np.random.uniform(ang_range)-ang_range/2
	rows,cols,ch = img.shape
	Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

	# Translation
	tr_x = trans_range*np.random.uniform()-trans_range/2
	tr_y = trans_range*np.random.uniform()-trans_range/2
	Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

	# Shear
	pts1 = np.float32([[5,5],[20,5],[5,20]])

	pt1 = 5+shear_range*np.random.uniform()-shear_range/2
	pt2 = 20+shear_range*np.random.uniform()-shear_range/2

	# Brightness


	pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

	shear_M = cv2.getAffineTransform(pts1,pts2)

	img = cv2.warpAffine(img,Rot_M,(cols,rows))
	img = cv2.warpAffine(img,Trans_M,(cols,rows))
	img = cv2.warpAffine(img,shear_M,(cols,rows))


	return img




print(max_count, train_hist)



def create_data():
	train_hist = np.bincount(y_train)
	max_count = np.max(train_hist)*1.5
	X_train_aug = []
	y_train_aug = []
	for i in range(len(y_train)):
		img = X_train[i]
		label = y_train[i]
		X_train_aug.append(img)
		y_train_aug.append(label)
		for e in range(math.floor(max_count/train_hist[label])):
			img_transformed = transform_image(img, 20, 10, 5)
			X_train_aug.append(img_transformed)
			y_train_aug.append(label)

	return np.array(X_train_aug), np.array(y_train_aug)



augment_file_test = 'train_aug.npy'

if not os.path.isfile(augment_file_test):
	X_train_aug, y_train_aug = create_data()
	X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug, random_state=0)

	print("Shape ", X_train_aug.shape, y_train_aug.shape)

	with open(augment_file_test, 'wb') as output:
		pickle.dump(X_train_aug, output, pickle.HIGHEST_PROTOCOL)
		pickle.dump(y_train_aug, output, pickle.HIGHEST_PROTOCOL)

with open(augment_file_test, 'rb') as input:
	X_train = pickle.load(input)
	y_train = pickle.load(input)


print("Shape ", X_train.shape, y_train.shape)


train_hist = np.bincount(y_train)
max_count = np.max(train_hist)*1.5

print(max_count, train_hist)

print(y_train[0:100])


import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.gridspec as gridspec
# Visualizations will be shown in the notebook.

imgs = X_train[np.random.randint(n_train, size=20)]

gs1 = gridspec.GridSpec(2, 10)
gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes.
plt.figure(figsize=(10,10))
for i in range(len(imgs)):
	ax1 = plt.subplot(gs1[i])
	ax1.set_xticklabels([])
	ax1.set_yticklabels([])
	ax1.set_aspect('equal')
	img = imgs[i]

	plt.subplot(2,10,i+1)
	plt.imshow(img)
	plt.axis('off')

plt.show()



