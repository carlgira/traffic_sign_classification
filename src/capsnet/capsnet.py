import src.traffic_sign_classifier as tsf
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from src.capsnet.capsLayer import CapsLayer

class CapsNet_Cardn(tsf.TrafficSignClassifier):

	def __init__(self, name):
		super().__init__(name)
		self.BATCH_SIZE = 128
		self.keep_rate_dense = 0.75

	# Neural Network
	def neural_network(self, x_data, y_data, phase):

		conv1 = tf.layers.conv2d(x_data, filters=32, kernel_size=[9,9], padding='VALID')

		conv1_size = int(32-9/1) + 1
		caps1_size = int((conv1_size-9)/2.0) + 1
		caps_neurons = caps1_size*caps1_size*8

		caps_layer = CapsLayer(conv_num_outputs=8, fcc_num_outputs=43, conv_vec_len=4, fcc_vec_len=4, caps_neurons=caps_neurons, batch_size=self.BATCH_SIZE)

		caps1 = caps_layer.conv_layer(conv1, kernel_size=9, stride=2)

		caps2 = caps_layer.fcc_layer(caps1)

		fc0 = flatten(caps2)
		fc1 = tf.layers.dense(fc0, units=1024)
		fc1 = tf.layers.dropout(fc1, rate=self.keep_rate_dense, training=phase)
        
		fc2 = tf.layers.dense(fc1, units=256)
		fc2 = tf.layers.dropout(fc2, rate=self.keep_rate_dense, training=phase)
		
		logits = tf.layers.dense(fc2, units=self.classes)

		accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_data, 1 ), tf.argmax(logits, 1)), tf.float32), name='accuracy')
		loss_operation = tf.losses.softmax_cross_entropy(onehot_labels=y_data, logits=logits)
		training_operation = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_operation)

		return accuracy_operation, training_operation, loss_operation


if __name__ == '__main__':
	nn = CapsNet_Cardn('capsnets')
	nn.train_nn()