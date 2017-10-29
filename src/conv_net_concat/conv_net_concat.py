import src.traffic_sign_classifier as tsf
import tensorflow as tf
from tensorflow.contrib.layers import flatten

class ConvNet(tsf.TrafficSignClassifier):

	def __init__(self, name):
		super().__init__(name)
		self.keep_rate_dense = 0.5

	# Neural Network
	def neural_network(self, x_data, y_data, phase):
		conv1 = tf.layers.conv2d(x_data, filters=8, kernel_size=[1, 1], padding='valid', activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
		conv1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)
		conv1 = tf.layers.batch_normalization(conv1, training=phase)

		conv2 = tf.layers.conv2d(conv1, filters=16, kernel_size=[1, 1], padding='valid', activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
		conv2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)
		conv2 = tf.layers.batch_normalization(conv2, training=phase)

		conv3 = tf.layers.conv2d(conv2, filters=32, kernel_size=[1, 1], padding='valid', activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
		conv3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2)
		conv3 = tf.layers.batch_normalization(conv3, training=phase)

		fc0 = tf.concat([flatten(conv3), flatten(conv2), flatten(conv1)], axis=1)

		fc1 = tf.layers.dense(fc0, units=512, activation=tf.nn.relu)
		fc1 = tf.layers.dropout(fc1, rate=self.keep_rate_dense, training=phase)
		fc1 = tf.layers.batch_normalization(fc1, training=phase)

		fc2 = tf.layers.dense(fc1, units=128, activation=tf.nn.relu)
		fc2 = tf.layers.dropout(fc2, rate=self.keep_rate_dense, training=phase)
		fc2 = tf.layers.batch_normalization(fc2, training=phase)

		logits = tf.layers.dense(fc2, units=self.classes)

		output = tf.nn.softmax(logits)

		accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y_data, 1 ), tf.arg_max(output, 1)), tf.float32), name='accuracy')
		loss_operation = tf.losses.softmax_cross_entropy(onehot_labels=y_data, logits=logits)
		training_operation = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_operation)

		return accuracy_operation, training_operation, loss_operation

if __name__ == '__main__':
	nn = ConvNet('conv_net_concat')
	nn.train_nn()
