from consts import *
from data import *
from utilits import *


class TwoLayerNetwork():

    def __init__(self, r ,lr, W_init=None, U_init=None, B_W_init=None, B_U_init=None, sigma_1=SIGMA, sigma_2=SIGMA):
        print("Initialize two layer network with gaussin initialization")
        if W_init is not None:
            self.W = W_init
            self.U = U_init
            self.B_W = B_W_init
            self.B_U = B_U_init
            self.r = W_init.shape[0]
        else:
            self.r = r 
            self.W = np.array(sigma_1 * np.random.randn(self.r, D), dtype=TYPE)
            self.U = np.array(sigma_2 * np.random.randn(self.r), dtype=TYPE)
            self.B_W = np.zeros([self.r], dtype=TYPE)
            self.B_U = np.zeros([1], dtype=TYPE)
        self.lr = lr


    def run(self, train_set, test_set):
        train_size = train_set[0].shape[0]
        test_size = test_set[0].shape[0]

        tf.reset_default_graph()
        with tf.Graph().as_default():
            # init variables
            global_step = tf.Variable(0, trainable=False, name='global_step')
            X = tf.placeholder(TYPE, name='X', shape=[None, D])
            Y = tf.placeholder(TYPE, name='Y', shape=[None])
            W = tf.get_variable('W', initializer=self.W.T)
            U = tf.get_variable('U', initializer=self.U)
            B_W = tf.get_variable('B_W', initializer=self.B_W)
            B_U = tf.get_variable('B_U', initializer=self.B_U)
            
            # Netrowk
            out_1 = tf.nn.relu(tf.matmul(X, W) + B_W)
            logits = tf.tensordot(out_1, U, 1) + B_U

            # calc accuracy
            prediction_train = tf.round(tf.nn.sigmoid(logits))
            ones_train = tf.constant(np.ones([train_size]), dtype=TYPE)
            zeros_train = tf.constant(np.zeros([train_size]), dtype=TYPE)
            correct_train = tf.where(tf.equal(prediction_train, Y), ones_train, zeros_train)
            accuracy_train = tf.reduce_mean(correct_train)
            
            prediction_test = tf.round(tf.nn.sigmoid(logits))
            ones_test = tf.constant(np.ones([test_size]), dtype=TYPE)
            zeros_test = tf.constant(np.zeros([test_size]), dtype=TYPE)
            correct_test = tf.where(tf.equal(prediction_test, Y), ones_test, zeros_test)
            accuracy_test = tf.reduce_mean(correct_test)

            # calc loss
            loss_vec = tf.losses.hinge_loss(logits=logits, labels=Y, reduction=tf.losses.Reduction.NONE)
            loss = tf.reduce_mean(loss_vec)
            
            # set optimizer
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            gradient = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(gradient, global_step=global_step)

            with tf.Session() as sess:
                # init params
                init = tf.initialize_all_variables()
                sess.run(init)
                
                # train
                for step in range(0, MAX_STEPS):
                    _, train_loss, train_acc, current_gradient = sess.run([train_op, loss, accuracy_train, gradient], {X:train_set[0], Y:shift_label(train_set[1])})
                    if (np.sum(np.abs(current_gradient[0][0])) == 0 and np.sum(np.abs(current_gradient[1][0])) == 0 and np.sum(np.abs(current_gradient[2][0])) == 0 and np.sum(np.abs(current_gradient[3][0])) == 0) or (train_loss == 0):
                        print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc)) 
                        break 
                    if step % PRINT_STEP_JUMP == 0:
                        print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc)) 
                print("NN Train accuracy: {0}".format(train_acc)) 
                
                test_loss, test_acc = sess.run([loss, accuracy_test], {X:test_set[0], Y:shift_label(test_set[1])})  
                print('NN Test accuracy: {0}'.format(test_acc)) 
                self.W, self.U, self.B_W, self.B_U = sess.run([W, U, B_W, B_U])
                self.W = self.W.T

            return train_loss, test_acc