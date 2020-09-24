from consts import *
from data import *
from utilits import *


class FixLayerTwoNetwork():

    def __init__(self, epsilon_init, lr, r=0, W_init=None, B_init=None, use_batch=False):
        assert epsilon_init or (not epsilon_init and r > 0) or (W_init is not None and B_init is not None)
        # init graph
        if epsilon_init:
            print("Initialize fix layer two network with epsilon initialization")
            all_combinations = get_all_combinations()
            if r > 0:
                self.W = np.array(get_random_init_uniform_samples(r), dtype=TYPE) * SIGMA
                self.r = r
            else:
                self.W = np.array(all_combinations, dtype=TYPE) * SIGMA
                self.r = 2 ** D
            self.W = self.W
            self.B = np.zeros([self.r], dtype=TYPE)
        elif r > 0:
            print("Initialize fix layer two network with gaussin initialization")
            self.r = r 
            self.W = np.array(SIGMA * np.random.randn(self.r, D), dtype=TYPE)
            self.B = np.zeros([self.r], dtype=TYPE)
        else:
            self.r = W_init.shape[0]
            self.W = W_init
            self.B = B_init
        self.lr = lr
        self.all_W = [self.W]
        self.all_B = [self.B]
        self.use_batch = use_batch

    def train_without_batch(self, train_set, sess):
        for step in range(0, MAX_STEPS):
            _, train_loss, train_acc, current_gradient = sess.run([self.train_op, self.loss, self.accuracy_train, self.gradient], {self.X:train_set[0], self.Y:shift_label(train_set[1])})
            if (np.sum(np.abs(current_gradient[0][0])) == 0 and np.sum(np.abs(current_gradient[1][0])) == 0) or (train_loss == 0):
                print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc)) 
                break 
            if step % PRINT_STEP_JUMP == 0:
                print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc)) 
        print("NN Train accuracy: {0}".format(train_acc)) 
        return train_loss

    def train_with_batch(self, train_set, sess):
        all_train_size = train_set[0].shape[0]
        step = 0
        while step < MAX_STEPS:
            for i in range(0, all_train_size, BATCH_SIZE):
                if i + BATCH_SIZE < all_train_size:
                    X_batch, Y_batch =  train_set[0][i : i + BATCH_SIZE], train_set[1][i : i + BATCH_SIZE]
                    _, train_loss, train_acc, current_gradient = sess.run([self.train_op, self.loss, self.accuracy_train, self.gradient], {self.X:X_batch, self.Y:shift_label(Y_batch)})
                else:
                    X_batch, Y_batch =  train_set[0][i:], train_set[1][i:]
                    _, train_loss, current_gradient = sess.run([self.train_op, self.loss, self.gradient], {self.X:X_batch, self.Y:shift_label(Y_batch)})
                    train_acc = -1
                step += 1
                if (np.sum(np.abs(current_gradient[0][0])) == 0 and np.sum(np.abs(current_gradient[1][0])) == 0) or (train_loss == 0):
                    print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc)) 
                    break 
                if step % PRINT_STEP_JUMP == 0:
                    print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc)) 
        print("NN Train accuracy: {0}".format(train_acc)) 
        return train_loss        

    def run(self, train_set, test_set, just_test=False):
        if self.use_batch:
            train_size = BATCH_SIZE
        else:
            train_size = train_set[0].shape[0]
        test_size = test_set[0].shape[0]
        train_loss, test_acc = 0, 0
        tf.reset_default_graph()
        with tf.Graph().as_default():
            # init variables
            global_step = tf.Variable(0, trainable=False, name='global_step')
            self.X = tf.placeholder(TYPE, name='X', shape=[None, D])
            self.Y = tf.placeholder(TYPE, name='Y', shape=[None])
            W = tf.get_variable('W', initializer=self.W.T)
            B = tf.get_variable('B_W', initializer=self.B)
            
            # Netrowk
            out_1 = tf.nn.relu(tf.matmul(self.X, W) + B)
            logits = tf.reduce_sum(out_1, axis=1)  - 1

            # calc accuracy
            prediction_train = tf.round(tf.nn.sigmoid(logits))
            ones_train = tf.constant(np.ones([train_size]), dtype=TYPE)
            zeros_train = tf.constant(np.zeros([train_size]), dtype=TYPE)
            correct_train = tf.where(tf.equal(prediction_train, self.Y), ones_train, zeros_train)
            self.accuracy_train = tf.reduce_mean(correct_train)
            
            prediction_test = tf.round(tf.nn.sigmoid(logits))
            ones_test = tf.constant(np.ones([test_size]), dtype=TYPE)
            zeros_test = tf.constant(np.zeros([test_size]), dtype=TYPE)
            correct_test = tf.where(tf.equal(prediction_test, self.Y), ones_test, zeros_test)
            accuracy_test = tf.reduce_mean(correct_test)

            # calc loss
            loss_vec = tf.losses.hinge_loss(logits=logits, labels=self.Y, reduction=tf.losses.Reduction.NONE)
            self.loss = tf.reduce_mean(loss_vec)
            
            # set optimizer
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.gradient = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(self.gradient, global_step=global_step)

            with tf.Session() as sess:
                # init params
                init = tf.initialize_all_variables()
                sess.run(init)
                
                # train
                if not just_test:
                    if self.use_batch:
                        train_loss = self.train_with_batch(train_set, sess)    
                    else:
                        train_loss = self.train_without_batch(train_set, sess)
                    
                test_loss, test_acc = sess.run([self.loss, accuracy_test], {self.X:test_set[0], self.Y:shift_label(test_set[1])})  
                print('NN Test accuracy: {0}'.format(test_acc)) 

                self.W, self.B = sess.run([W, B])
                self.W = self.W.T

            return train_loss, test_acc