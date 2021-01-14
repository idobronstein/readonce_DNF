from consts import *
from data import *
from utilits import *


class NTKsvn():

	def __init__(self, r):
		self.W = np.array(SIGMA * np.random.randn(r, D), dtype=TYPE)

	def calc_position_vector(self, x):
		Wx = tf.linalg.matvec(self.W, x)
		Wx = tf.where(Wx < 0, tf.zeros(Wx.shape, dtype=TYPE), tf.ones(Wx.shape, dtype=TYPE))
		return Wx

	def prepere_mult_by_kernel(self):
		self.x = tf.placeholder(TYPE, name='x', shape=[D])
		self.y = tf.placeholder(TYPE, name='x', shape=[D])
		dot_product = tf.tensordot(self.x, self.y, 1)
		Wx = self.calc_position_vector(self.x)
		Wy = self.calc_position_vector(self.y)
		return dot_product * tf.tensordot(Wx, Wy, 1)
		
	def calc_distance_matrix(self, samples1, samples2, sess, tf_distance_matrix):
		N = samples1.shape[0]
		M = samples2.shape[0]
		distance_matrix = np.zeros([N, M])
		for i in range(N):
			for j in range(M):
				distance_matrix[i][j] = sess.run([tf_distance_matrix], {self.x: samples1[i], self.y: samples2[j]})[0]
		return distance_matrix

	def run(self, train_set, test_set):
		tf.reset_default_graph()
		with tf.Graph().as_default():
			tf_distance_matrix = self.prepere_mult_by_kernel()
			with tf.Session() as sess:
				# init params
				init = tf.initialize_all_variables()
				sess.run(init)
				
				train_distance_matrix = self.calc_distance_matrix(train_set[0], train_set[0], sess, tf_distance_matrix)
				test_distance_matrix = self.calc_distance_matrix(test_set[0], train_set[0], sess, tf_distance_matrix)
		
		learner = SVC(kernel='precomputed')
		learner.fit(train_distance_matrix, train_set[1])
		
		train_accuracy = np.dot(np.sign(learner.predict(train_distance_matrix)), train_set[1]) / train_set[0].shape[0]
		print("NTK Train accuracy: {0}".format(train_accuracy))
		
		test_accuracy = np.dot(np.sign(learner.predict(test_distance_matrix)), test_set[1]) / test_set[0].shape[0]
		print("NTK Test accuracy: {0}".format(test_accuracy)) 