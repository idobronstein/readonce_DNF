from consts import *
from data import *
from utilits import *


class NTKsvn():

	def __init__(self, r):
		self.W = np.array(SIGMA * np.random.randn(r, D), dtype=TYPE)

	def calc_position_matrix(self, X):
		XW = tf.tensordot(X, tf.transpose(self.W), 1)
		XW = tf.where(XW < 0, tf.zeros(XW.shape, dtype=TYPE), tf.ones(XW.shape, dtype=TYPE))
		return XW

	def prepere_distance_matrix(self, x_set_size, y_set_size):
		self.X = tf.placeholder(TYPE, name='X', shape=[x_set_size, D])
		self.Y = tf.placeholder(TYPE, name='Y', shape=[y_set_size, D])
		XW = self.calc_position_matrix(self.X)
		YW = self.calc_position_matrix(self.Y)
		Wxy = tf.tensordot(XW, tf.transpose(YW), 1)
		dot_product_matrix = tf.tensordot(self.X, tf.transpose(self.Y), 1)		
		tf_distance_matrix = tf.multiply(dot_product_matrix, Wxy)
		return tf_distance_matrix

	def run(self, train_set, test_set):
		train_set_size = train_set[0].shape[0]
		test_set_size = test_set[0].shape[0]
		tf.reset_default_graph()
		with tf.Graph().as_default():
			tf_distance_matrix = self.prepere_distance_matrix(train_set_size, train_set_size)
			with tf.Session() as sess:
				# init params
				init = tf.initialize_all_variables()
				sess.run(init)
				
				train_distance_matrix = sess.run([tf_distance_matrix], {self.X: train_set[0], self.Y: train_set[0]})[0]
				
		
		learner = SVC(kernel='precomputed')
		learner.fit(train_distance_matrix, train_set[1])
		
		train_accuracy = np.dot(np.sign(learner.predict(train_distance_matrix)), train_set[1]) / train_set[0].shape[0]
		print("NTK Train accuracy: {0}".format(train_accuracy))
		
		tf.reset_default_graph()
		with tf.Graph().as_default():
			tf_distance_matrix = self.prepere_distance_matrix(test_set_size, train_set_size)
			with tf.Session() as sess:
				# init params
				init = tf.initialize_all_variables()
				sess.run(init)

				test_distance_matrix = sess.run([tf_distance_matrix], {self.X: test_set[0], self.Y: train_set[0]})[0]

		test_accuracy = np.dot(np.sign(learner.predict(test_distance_matrix)), test_set[1]) / test_set[0].shape[0]
		print("NTK Test accuracy: {0}".format(test_accuracy)) 