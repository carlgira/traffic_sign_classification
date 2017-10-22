import pickle
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import os.path
import math
import numpy as np
import cv2


def load_dataset(training_file, validation_file, testing_file):

	with open(training_file, mode='rb') as f:
		train = pickle.load(f)
	with open(validation_file, mode='rb') as f:
		valid = pickle.load(f)
	with open(testing_file, mode='rb') as f:
		test = pickle.load(f)

	return train, valid, test

train, valid, test = load_dataset("traffic-signs-data/train.p", "traffic-signs-data/test.p", "traffic-signs-data/valid.p")

X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


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


EPOCHS = 10
BATCH_SIZE = 128
channels = X_train.shape[3]
rate = 0.001
classes = 43

x = tf.placeholder(tf.float32, (None, 32, 32, channels))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, classes)

conv1 = tf.layers.conv2d(x, filters=16, kernel_size=[5, 5], padding='valid', activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
conv1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2)
conv1 = tf.layers.dropout(conv1, rate=0.25)

conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=[5,5], padding='valid', activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
conv2 = tf.layers.max_pooling2d(conv2, pool_size=[2,2], strides=2)
conv2 = tf.layers.dropout(conv2, rate=0.25)

fc0 = flatten(conv2)
fc1 = tf.layers.dense(fc0, units=256, activation=tf.nn.relu)
fc1 = tf.layers.dropout(fc1, rate=0.5)

fc2 = tf.layers.dense(fc1, units=128, activation=tf.nn.relu)
fc2 = tf.layers.dropout(fc2, rate=0.5)

logits = tf.layers.dense(fc2, units=classes)

output = tf.nn.softmax(logits)

accuracy_operation = tf.reduce_mean(tf.cast( tf.equal(tf.arg_max(one_hot_y,1 ), tf.arg_max(output, 1)), tf.float32))
loss_operation = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_y, logits=logits)
training_operation = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss_operation)


saver = tf.train.Saver()

def evaluate(X_data, y_data):
	num_examples = len(X_data)
	total_accuracy = 0
	sess = tf.get_default_session()
	for offset in range(0, num_examples, BATCH_SIZE):
		batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
		accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
		total_accuracy += (accuracy * len(batch_x))
	return total_accuracy / num_examples


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	num_examples = len(X_train)

	print("Training...")
	print()
	for i in range(EPOCHS):
		X_train, y_train = shuffle(X_train, y_train)
		for offset in range(0, num_examples, BATCH_SIZE):
			end = offset + BATCH_SIZE
			batch_x, batch_y = X_train[offset:end], y_train[offset:end]
			sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

		validation_accuracy = evaluate(X_validation, y_validation)
		print("EPOCH {} ...".format(i+1))
		print("Validation Accuracy = {:.3f}".format(validation_accuracy))
		print()

	saver.save(sess, 'lenet')
	print("Model saved")

#with tf.Session() as sess:
#	saver.restore(sess, tf.train.latest_checkpoint('.'))

#	test_accuracy = evaluate(X_test, y_test)
#	print("Test Accuracy = {:.3f}".format(test_accuracy))