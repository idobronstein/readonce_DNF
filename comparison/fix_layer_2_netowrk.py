from consts import *
from data import *
from utilits import *


class FixLayerTwoNetwork():

    def __init__(self, epsilon_init, lr, r=0):
        assert epsilon_init or (not epsilon_init and r > 0)
        # init graph
        if epsilon_init:
            print("Initialize fix layer two network with epsilon initialization")
            self. r = 2 ** D
            self.W_init = np.array(get_all_combinations(), dtype=FLOAT_TYPE) * SIGAM_EPSILON
            self.W_init = self.W_init.T
        else:
            print("Initialize fix layer two network with gaussin initialization")
            self.r = r 
            self.W_init = np.array(SIGMA_GAUSS * np.random.randn(D, self.r), dtype=FLOAT_TYPE)
        self.B_init = np.zeros([self.r], dtype=FLOAT_TYPE)
        self.lr = lr

    def run(self, train_set, test_set):
        train_size = train_set[0].shape[0]
        test_size = test_set[0].shape[0]

        tf.reset_default_graph()
        with tf.Graph().as_default():
            # init variables
            global_step = tf.Variable(0, trainable=False, name='global_step')
            X = tf.placeholder(FLOAT_TYPE, name='X', shape=[None, D])
            Y = tf.placeholder(FLOAT_TYPE, name='Y', shape=[None])
            W = tf.get_variable('W', initializer=self.W_init)
            B = tf.get_variable('B_W', initializer=self.B_init)
            
            # Netrowk
            out_1 = tf.nn.relu(tf.matmul(X, W) + B)
            logits = tf.reduce_sum(out_1, axis=1)  - 1

            # calc accuracy
            prediction_train = tf.round(tf.nn.sigmoid(logits))
            ones_train = tf.constant(np.ones([train_size]), dtype=FLOAT_TYPE)
            zeros_train = tf.constant(np.zeros([train_size]), dtype=FLOAT_TYPE)
            correct_train = tf.where(tf.equal(prediction_train, Y), ones_train, zeros_train)
            accuracy_train = tf.reduce_mean(correct_train)
            
            prediction_test = tf.round(tf.nn.sigmoid(logits))
            ones_test = tf.constant(np.ones([test_size]), dtype=FLOAT_TYPE)
            zeros_test = tf.constant(np.zeros([test_size]), dtype=FLOAT_TYPE)
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
                    _, train_loss, train_acc = sess.run([train_op, loss, accuracy_train], {X:train_set[0], Y:shift_label(train_set[1])})
                    if train_loss == 0:
                        print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc)) 
                        break 
                    if step % PRINT_STEP_JUMP == 0:
                        print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc)) 
                print("NN Train accuracy: {0}".format(train_acc)) 
                
                test_loss, test_acc = sess.run([loss, accuracy_test], {X:test_set[0], Y:shift_label(test_set[1])})  
                print('NN Test accuracy: {0}'.format(test_acc)) 

            return test_acc