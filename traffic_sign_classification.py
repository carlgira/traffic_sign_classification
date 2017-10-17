import pickle
import tensorflow as tf
import numpy as np


def download_dataset():
	pass
	# https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
	return


def load_dataset(training_file, validation_file, testing_file):

	with open(training_file, mode='rb') as f:
		train = pickle.load(f)
	with open(validation_file, mode='rb') as f:
		valid = pickle.load(f)
	with open(testing_file, mode='rb') as f:
		test = pickle.load(f)

	return train, valid, test


download_dataset()
train, valid, test = load_dataset("traffic-signs-data/train.p", "traffic-signs-data/test.p", "traffic-signs-data/valid.p")

x_train, y_train = train['features'], train['labels']
x_valid, y_valid = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']

n_train = x_train.shape[0]
batch_size = 10
learning_rate = 0.001
training_epochs = 10
img_size = 32
n_classes = 43


X = tf.placeholder(dtype=tf.float32, shape=[None, img_size, img_size, 3])
Y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])

conv1 = tf.layers.conv2d(X, filters=32, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
maxp1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2)

conv2 = tf.layers.conv2d(maxp1, filters=64, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
maxp2 = tf.layers.max_pooling2d(conv2, pool_size=[2,2], strides=2)

conv3 = tf.layers.conv2d(maxp2, filters=128, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
maxp3 = tf.layers.max_pooling2d(conv3, pool_size=[2,2], strides=2)


fc_layer = tf.reshape(maxp3, [-1, 4*4*128])
fc_layer1 = tf.layers.dense(fc_layer, units=1000, activation=tf.nn.relu)

fc_layer2 = tf.layers.dense(fc_layer1, units=500, activation=tf.nn.relu)

logits = tf.layers.dense(fc_layer2, units=n_classes)

output = tf.nn.softmax(logits)

accuracy = tf.reduce_mean(tf.cast( tf.equal(tf.arg_max(Y,1 ), tf.arg_max(output, 1)), tf.float32))

loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=output)

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())


	test_x, test_y = x_test, np.eye(n_classes)[y_test.reshape(-1)]

	for epoch in range(training_epochs):
		#batch_count = int(n_train/batch_size)
		batch_count = 100
		for i in range(batch_count):
			batch_x, batch_y = x_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]
			batch_y = np.eye(n_classes)[batch_y]

			#print(sess.run([tf.shape(maxp3)], feed_dict={X: batch_x, Y: batch_y}))
			print(sess.run([t1, t2], feed_dict={X: batch_x, Y: batch_y}))
			sess.run([train_step], feed_dict={X: batch_x, Y: batch_y})
			print("Accuracy", sess.run([accuracy], feed_dict={X: batch_x, Y: batch_y}))

		#print("Accuracy", sess.run([accuracy, accuracy2], feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
		print("Accuracy Test", sess.run([accuracy], feed_dict={X: test_x, Y: test_y}))
