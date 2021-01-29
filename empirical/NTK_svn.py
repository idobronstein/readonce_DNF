from consts import *
from data import *
from utilits import *


class NTKsvn():

	def __init__(self, r):
		self.r = r
		self.W = np.array(SIGMA * np.random.randn(r, D+1), dtype=TYPE)

	def calc_position_matrix(self, X):
		XW = tf.tensordot(X, tf.transpose(self.W), 1)
		XW = tf.where(XW < 0, tf.zeros(XW.shape, dtype=TYPE), tf.ones(XW.shape, dtype=TYPE))
		return XW

	def prepere_distance_matrix_1(self, x_set_size, y_set_size):
		self.X1 = tf.placeholder(TYPE, name='X1', shape=[x_set_size, D + 1])
		self.Y1 = tf.placeholder(TYPE, name='Y1', shape=[y_set_size, D + 1])
		XW = self.calc_position_matrix(X1)
		YW = self.calc_position_matrix(self.Y1)
		Wxy = tf.tensordot(XW, tf.transpose(YW), 1)
		dot_product_matrix = tf.tensordot(X1, tf.transpose(Y1), 1)		
		tf_distance_matrix = (1 / self.r) * tf.multiply(dot_product_matrix, Wxy)
		return tf_distance_matrix


	def prepere_distance_matrix_2(self, x_set_size, y_set_size):
		self.X2 = tf.placeholder(TYPE, name='X2', shape=[x_set_size, D + 1])
		self.Y2 = tf.placeholder(TYPE, name='Y2', shape=[y_set_size, D + 1])
		dot_product_matrix = my_tf_round(tf.tensordot(self.X2, tf.transpose(self.Y2), 1), 6)	
		tf_distance_matrix = tf.multiply((1 / (2 * math.pi)) * (dot_product_matrix), (math.pi - tf.math.acos(dot_product_matrix)))
		return tf_distance_matrix

	def run(self, train_set, test_set):
		train_set_size = train_set[0].shape[0]
		test_set_size = test_set[0].shape[0]
		train_scale = np.concatenate([train_set[0], np.ones([train_set_size, 1])], 1) / np.sqrt(D + 1)
		test_scale = np.concatenate([test_set[0], np.ones([test_set_size, 1])], 1) / np.sqrt(D + 1)
		tf.reset_default_graph()
		with tf.Graph().as_default():
			tf_distance_matrix = self.prepere_distance_matrix_2(train_set_size, train_set_size)
			with tf.Session() as sess:
				# init params
				init = tf.initialize_all_variables()
				sess.run(init)
				train_distance_matrix = sess.run([tf_distance_matrix], {self.X2: train_scale, self.Y2: train_scale})[0]

		learner = SVC(kernel='precomputed')
		learner.fit(train_distance_matrix, train_set[1])
		
		train_accuracy = np.dot(np.sign(learner.predict(train_distance_matrix_1)), train_set[1]) / train_set_size
		print("NTK Train accuracy: {0}".format(train_accuracy))
		
		tf.reset_default_graph()
		with tf.Graph().as_default():
			tf_distance_matrix = self.prepere_distance_matrix_2(test_set_size, train_set_size)
			with tf.Session() as sess:
				# init params
				init = tf.initialize_all_variables()
				sess.run(init)
				test_distance_matrix = sess.run([tf_distance_matrix], {self.X2: test_scale, self.Y2: train_scale})[0]

		test_accuracy = np.dot(np.sign(learne.predict(test_distance_matrix)), test_set[1]) / test_set_size
		print("NTK Test accuracy: {0}".format(test_accuracy)) 

		return 1 - train_accuracy, test_accuracy