import tensorflow as tf
import numpy as np

def yololike_model(features, labels, mode):
	"""YOLO-like CNN architecture.
	Parameters:
		features: the array containing the examples used for training
		labels: the array containing the labels of the examples in one-hot representation
		mode: a tf.estimator.ModeKeys like tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.PREDICT
	"""

	if features.shape[1].value < 100:
		one_dim_pool = True
	else:
		one_dim_pool = False

	# Input Layer
	input_layer = tf.reshape(features, [-1, features.shape[1], features.shape[2], 1])
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=64,
		kernel_size=[7, 7],
		strides=2,
		padding="same",
		activation=tf.nn.relu)
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=192,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	conv3 = tf.layers.conv2d(
		inputs=pool2,
		filters=128,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)

	conv4 = tf.layers.conv2d(
		inputs=conv3,
		filters=256,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	conv5 = tf.layers.conv2d(
		inputs=conv4,
		filters=256,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)

	conv6 = tf.layers.conv2d(
		inputs=conv5,
		filters=512,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	if one_dim_pool:
		pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[1, 2], strides=2)
	else:
		pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

	conv7 = tf.layers.conv2d(
		inputs=pool3,
		filters=256,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)
	conv8 = tf.layers.conv2d(
		inputs=conv7,
		filters=512,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)
	conv9 = tf.layers.conv2d(
		inputs=conv8,
		filters=1024,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	if one_dim_pool:
		pool4 = tf.layers.max_pooling2d(inputs=conv9, pool_size=[1, 2], strides=2)
	else:
		pool4 = tf.layers.max_pooling2d(inputs=conv9, pool_size=[2, 2], strides=2)


	conv10 = tf.layers.conv2d(
		inputs=pool4,
		filters=1024,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	print('conv10', conv10.shape)

	pool4_flat = tf.reshape(conv10, [-1, np.prod(conv10.shape[1:])])
	dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
	dropout = tf.layers.dropout(inputs=dense, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits Layer
	logits = tf.layers.dense(inputs=dropout, units=labels.shape[1])

	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=tf.argmax(input=labels, axis=1),
			predictions=predictions["classes"]
		)
	}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def vgg16_model(features, labels, mode):
	"""VGG-16 CNN architecture.
	Parameters:
		features: the array containing the examples used for training
		labels: the array containing the labels of the examples in one-hot representation
		mode: a tf.estimator.ModeKeys like tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.PREDICT
	"""
	if features.shape[1].value < 100:
		one_dim_pool = True
	else:
		one_dim_pool = False

	# Input Layer
	input_layer = tf.reshape(features, [-1, features.shape[1], features.shape[2], 1])
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=64,
		kernel_size=[3, 3],
		strides=1,
		padding="same",
		activation=tf.nn.relu)
	conv2 = tf.layers.conv2d(
		inputs=conv1,
		filters=64,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	if one_dim_pool:
		pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 2], strides=2)
	else:
		pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	conv3 = tf.layers.conv2d(
		inputs=pool1,
		filters=128,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	conv4 = tf.layers.conv2d(
		inputs=conv3,
		filters=128,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	if one_dim_pool:
		pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[1, 2], strides=2)
	else:
		pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

	conv5 = tf.layers.conv2d(
		inputs=pool2,
		filters=256,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)
	conv6 = tf.layers.conv2d(
		inputs=conv5,
		filters=256,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)
	conv7 = tf.layers.conv2d(
		inputs=conv6,
		filters=256,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)

	if one_dim_pool:
		pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[1, 2], strides=2)
	else:
		pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)

	conv8 = tf.layers.conv2d(
		inputs=pool3,
		filters=512,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)
	conv9 = tf.layers.conv2d(
		inputs=conv8,
		filters=512,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)
	conv10 = tf.layers.conv2d(
		inputs=conv9,
		filters=512,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)

	if one_dim_pool:
		pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[1, 2], strides=2)
	else:
		pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)

	conv11 = tf.layers.conv2d(
		inputs=pool4,
		filters=512,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)
	conv12 = tf.layers.conv2d(
		inputs=conv11,
		filters=512,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)
	conv13 = tf.layers.conv2d(
		inputs=conv12,
		filters=512,
		kernel_size=[1, 1],
		padding="same",
		activation=tf.nn.relu)

	if one_dim_pool:
		pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[1, 2], strides=2)
	else:
		pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2)

	pool5_flat = tf.reshape(pool5, [-1, np.prod(pool5.shape[1:])])

	dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)
	dense2 = tf.layers.dense(inputs=dense1, units=4096, activation=tf.nn.relu)

	# Logits Layer
	logits = tf.layers.dense(inputs=dense2, units=labels.shape[1])

	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=tf.argmax(input=labels, axis=1),
			predictions=predictions["classes"]
		)
	}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
