import src.traffic_sign_classifier as tsf
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from src.capsnet.capsLayer import CapsLayer
import src.capsnet.config as cfg

class CapsNet_Cardn(tsf.TrafficSignClassifier):

	def __init__(self, name):
		super().__init__(name, data_aug=False)

	# Neural Network
	def neural_network(self, x_data, y_data, phase):

		conv1 = tf.contrib.layers.conv2d(x_data, num_outputs=8,
									 kernel_size=9, stride=1,
									 padding='VALID')

		conv1_size = int(32-9/1) + 1
		print(conv1_size)

		#assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]
		# (?, 24, 24, 256)
		print("conv1", conv1.get_shape())

		caps1_size = int((conv1_size-9)/2.0) + 1
		caps_neurons = caps1_size*caps1_size*8

		primaryCaps = CapsLayer(num_outputs=8, conv_vec_len=4, fcc_vec_len=16, caps_neurons=caps_neurons, with_routing=False, layer_type='CONV')
		caps1 = primaryCaps(conv1, kernel_size=9, stride=2)

		#assert caps1.get_shape() == [cfg.batch_size, 1152, 8, 1]
		print("caps1.shape", caps1.get_shape())

		digitCaps = CapsLayer(num_outputs=10, conv_vec_len=4, fcc_vec_len=16, caps_neurons=caps_neurons, with_routing=True, layer_type='FC')
		caps2 = digitCaps(caps1)

		fc = flatten(caps2)
		fc1 = tf.layers.dense(fc, units=100, activation=tf.nn.relu)
		fc2 = tf.layers.dense(fc1, units=self.classes)
		logits = tf.nn.softmax(fc2)

		accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y_data, 1 ), tf.arg_max(logits, 1)), tf.float32), name='accuracy')
		loss_operation = tf.losses.softmax_cross_entropy(onehot_labels=y_data, logits=logits)
		training_operation = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_operation)

		return accuracy_operation, training_operation, loss_operation


if __name__ == '__main__':
	nn = CapsNet_Cardn('capsnets')
	nn.train_nn()