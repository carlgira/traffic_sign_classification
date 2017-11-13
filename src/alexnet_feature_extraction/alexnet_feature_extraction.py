import src.traffic_sign_classifier as tsf
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import src.alexnet_feature_extraction.alexnet as alexnet

class Alexnet_FeatureExtraction(tsf.TrafficSignClassifier):

	def __init__(self, name):
		super().__init__(name)

	# Neural Network
	def neural_network(self, x_data, y_data, phase):

		resized = tf.image.resize_images(x_data, [227, 227])

		fc7 = alexnet.AlexNet(resized, feature_extract=True)
		fc7 = tf.stop_gradient(fc7)

		logits = tf.layers.dense(fc7, units=self.classes)

		accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y_data, 1 ), tf.arg_max(logits, 1)), tf.float32), name='accuracy')
		loss_operation = tf.losses.softmax_cross_entropy(onehot_labels=y_data, logits=logits)
		training_operation = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_operation)

		return accuracy_operation, training_operation, loss_operation

if __name__ == '__main__':
	nn = Alexnet_FeatureExtraction('alexnet_fe')
	nn.train_nn()
