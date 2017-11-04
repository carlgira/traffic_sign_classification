import pickle
import tensorflow as tf
from sklearn.utils import shuffle
import math
import numpy as np
import cv2
import abc


class TrafficSignClassifier:

	def __init__(self, name, data_aug=True):
		# Parametrizartion
		self.learning_rate = 0.001
		self.classes = 43
		self.EPOCHS = 80
		self.BATCH_SIZE = 128
		self.saver = None
		self.name = name

		train, valid, test = self.load_dataset("../../traffic-signs-data/train.p", "../../traffic-signs-data/test.p", "../../traffic-signs-data/valid.p")
		self.X_train, self.y_train = train['features'], train['labels']
		self.X_validation, self.y_validation = valid['features'], valid['labels']
		self.X_test, self.y_test = test['features'], test['labels']

		if data_aug:
			self.X_train, self.y_train = self.create_data()

		self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=0)

		self.channels = self.X_train.shape[3]

	def load_dataset(self, training_file, validation_file, testing_file):
		'''
		Function to load the dataset
		:param training_file:
		:param validation_file:
		:param testing_file:
		:return:
		'''
		with open(training_file, mode='rb') as f:
			train = pickle.load(f)
		with open(validation_file, mode='rb') as f:
			valid = pickle.load(f)
		with open(testing_file, mode='rb') as f:
			test = pickle.load(f)

		return train, valid, test
    

    def eq_Hist(self, img):
        #Histogram Equalization
        img2=img.copy() 
        img2[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img2[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        img2[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
        return img2

    def scale_img(self, img):
        img2=img.copy()
        sc_y=0.4*np.random.rand()+1.0
        img2=cv2.resize(img, None, fx=1, fy=sc_y, interpolation = cv2.INTER_CUBIC)
        c_x,c_y, sh = int(img2.shape[0]/2), int(img2.shape[1]/2), int(img_size/2)
        return img2


    def rotate_img(self, img):
        c_x,c_y = int(img.shape[0]/2), int(img.shape[1]/2)
        ang = 30.0*np.random.rand()-15
        Mat = cv2.getRotationMatrix2D((c_x, c_y), ang, 1.0)
        return cv2.warpAffine(img, Mat, img.shape[:2])

    def sharpen_img(self, img):
        gb = cv2.GaussianBlur(img, (5,5), 20.0)
        return cv2.addWeighted(img, 2, gb, -1, 0)
    #Compute linear image transformation ing*s+m
    def lin_img(img,s=1.0,m=0.0):
        img2=cv2.multiply(img, np.array([s]))
        return cv2.add(img2, np.array([m]))

    #Change image contrast; s>1 - increase
    def contr_img(self, img, s=1.0):
        m=127.0*(1.0-s)
        return lin_img(img, s, m)

    def transform_img(self, img):
        img2=sharpen_img(img)
        img2=contr_img(img2, 1.5)
        return eq_Hist(img2)

    def augment_img(self, img):
        img=contr_img(img, 1.8*np.random.rand()+0.2)
        img=rotate_img(img)
        img=scale_img(img)
        return transform_img(img)    
    

	def transform_image(self, img,ang_range,shear_range,trans_range):
		'''
		Function for data augmentation
		:param img: Image
		:param ang_range: Range of angles for rotation
		:param shear_range: Range of values to apply affine transform to
		:param trans_range: Range of values to apply translations over.
		:return:
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

	@abc.abstractmethod
	def neural_network(self, x_data, y_data, phase):
		pass

	def create_data(self):
		'''
		Method for data augmentation
		:return:
		'''
		train_hist = np.bincount(self.y_train)
		max_count = np.max(train_hist)*1.5
		X_train_aug = []
		y_train_aug = []
		for i in range(len(self.y_train)):
			img = self.X_train[i]
			label = self.y_train[i]
			X_train_aug.append(img)
			y_train_aug.append(label)
			for e in range(math.floor(max_count/train_hist[label])):
				img_transformed = self.augment_img(img)
				X_train_aug.append(img_transformed)
				y_train_aug.append(label)

		return np.array(X_train_aug), np.array(y_train_aug)

	def evaluate(self, X_data, y_data, accuracy_operation):
		'''
		Function to get accuracy of input data
		'''
		num_examples = len(X_data)
		total_accuracy = 0
		sess = tf.get_default_session()
		for offset in range(0, num_examples, self.BATCH_SIZE):
			batch_x, batch_y = X_data[offset:offset + self.BATCH_SIZE], y_data[offset:offset + self.BATCH_SIZE]
			accuracy = sess.run(accuracy_operation, feed_dict={'x:0': batch_x, 'y:0': batch_y, 'phase:0': False})
			total_accuracy += (accuracy * len(batch_x))
		return total_accuracy / num_examples

	def train_nn(self):
		# Neural Netowrk training
		x = tf.placeholder(tf.float32, (None, 32, 32, self.channels), name='x')
		y = tf.placeholder(tf.int32, (None), name='y')
		one_hot_y = tf.one_hot(y, self.classes)
		phase = tf.placeholder(tf.bool, name='phase')

		accuracy_operation, training_operation, loss_operation = self.neural_network(x, one_hot_y, phase)
		self.saver = tf.train.Saver()

		tf.summary.scalar("loss", loss_operation)
		tf.summary.scalar("accuracy", accuracy_operation)

		merged_summary_op = tf.summary.merge_all()

		summary_writer = tf.summary.FileWriter(self.name + '/logs', graph=tf.get_default_graph())

		# Neural Network training
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			num_examples = len(self.X_train)
			print("Training...")
			for i in range(self.EPOCHS):
				for offset in range(0, num_examples, self.BATCH_SIZE):
					end = offset + self.BATCH_SIZE
					batch_x, batch_y = self.X_train[offset:end], self.y_train[offset:end]
					_, summary = sess.run([training_operation, merged_summary_op], feed_dict={x: batch_x, y: batch_y, phase: True})

					summary_writer.add_summary(summary, i * num_examples + offset)

				validation_accuracy = self.evaluate(self.X_validation, self.y_validation, accuracy_operation)
				print("EPOCH {}, Validation Accuracy = {:.3f}".format(i+1, validation_accuracy))

			self.saver.save(sess, self.name + '/nn')
			print("Model saved")

		# Neural Netowrk test
		with tf.Session() as sess:
			self.saver.restore(sess, tf.train.latest_checkpoint(self.name))

			test_accuracy = self.evaluate(self.X_test, self.y_test, accuracy_operation)
			print("Test Accuracy = {:.3f}".format(test_accuracy))
