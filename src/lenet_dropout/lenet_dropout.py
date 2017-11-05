import src.traffic_sign_classifier as tsf
import tensorflow as tf
from tensorflow.contrib.layers import flatten

class LeNETDroupOut(tsf.TrafficSignClassifier):

	def __init__(self, name, data_aug=False):
		super().__init__(name, data_aug)
		self.keep_rate_dense = 0.75

	# Neural Network
	def neural_network(self, x_data, y_data, phase):
		conv1 = tf.layers.conv2d(x_data, filters=6, kernel_size=[5, 5], padding='valid', activation=tf.nn.relu)
		conv1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

		conv2 = tf.layers.conv2d(conv1, filters=16, kernel_size=[5, 5], padding='valid', activation=tf.nn.relu)
		conv2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

		fc0 = flatten(conv2)
		fc1 = tf.layers.dense(fc0, units=120)
		fc1 = tf.layers.dropout(fc1, rate=self.keep_rate_dense, training=phase)

		fc2 = tf.layers.dense(fc1, units=84)

		logits = tf.layers.dense(fc2, units=self.classes)

		accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y_data, 1 ), tf.arg_max(logits, 1)), tf.float32), name='accuracy')
		loss_operation = tf.losses.softmax_cross_entropy(onehot_labels=y_data, logits=logits)
		training_operation = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_operation)

		return accuracy_operation, training_operation, loss_operation

if __name__ == '__main__':
	nn = LeNETDroupOut('lenet_dropout')
	nn.train_nn()
