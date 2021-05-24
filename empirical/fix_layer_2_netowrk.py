from consts import *
from data import *
from utilits import *


class FixLayerTwoNetwork():

    def __init__(self, epsilon_init, lr, r=0, W_init=None, B_init=None, B0_init=None, use_batch=False, use_crossentropy=True, xavier_init=False, sigma=SIGMA, without_B0=False):
        assert epsilon_init or (not epsilon_init and r > 0) or (W_init is not None and B_init is not None)
        # init graph
        if epsilon_init:
            print("Initialize fix layer two network with epsilon initialization")
            all_combinations = get_all_combinations()
            if r > 0:
                self.W = np.array(get_random_init_uniform_samples(r), dtype=TYPE) * sigma
                self.r = r
            else:
                self.W = np.array(all_combinations, dtype=TYPE) * sigma
                self.r = 2 ** D
            self.W = self.W
            self.B = np.zeros([self.r], dtype=TYPE)
        elif r > 0:
            print("Initialize fix layer two network with gaussin initialization")
            self.r = r 
            if xavier_init:
                self.W = None
            else:
                self.W = np.array(sigma * np.random.randn(self.r, D), dtype=TYPE)
            self.B = np.zeros([self.r], dtype=TYPE) 
        else:
            self.r = W_init.shape[0]
            self.W = W_init
            self.B = B_init
        if B0_init is None:
            self.B0 = np.zeros([1], dtype=TYPE)
        else:
            self.B0 = B0_init
        self.lr = lr
        self.all_W = [self.W]
        self.all_B = [self.B]
        self.all_B0 = [self.B0]
        self.use_batch = use_batch
        self.use_crossentropy = use_crossentropy
        self.without_B0 = without_B0

    def train_without_batch(self, train_set, sess, result_object=None):
        X_positive = train_set[0][[i for i in range(train_set[0].shape[0]) if train_set[1][i] == POSITIVE]]
        Y_positive = np.ones([X_positive.shape[0]], dtype=TYPE)
        for step in range(0, MAX_STEPS):
            _, train_loss, train_acc, current_gradient = sess.run([self.train_op, self.loss, self.accuracy_train, self.gradient], {self.X:train_set[0], self.Y:shift_label(train_set[1])})
            #W, B, B0 = sess.run([self.W_tf, self.B_tf, self.B0_tf])
            #self.all_W.append(W.T)
            #self.all_B.append(B)
            #self.all_B0.append(B0)
            if (np.sum(np.abs(current_gradient[0][0])) == 0 and np.sum(np.abs(current_gradient[1][0])) == 0) or ((self.use_crossentropy and train_loss <= CROSSENTROPY_THRESHOLD) or (not self.use_crossentropy and train_loss <= 0)):
                print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc)) 
                break 
            if step % PRINT_STEP_JUMP == 0:
                print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc)) 
            if result_object:
                result_object.cluster_graph(self, "{0}".format(step))
        print("NN Train accuracy: {0}".format(train_acc)) 
        return train_loss

    def train_with_batch(self, train_set, sess, result_object=None):
        all_train_size = train_set[0].shape[0]
        step = 0
        while step < MAX_STEPS:
            minimum_point_counter = 0
            shuffle_indexes = range(all_train_size)
            X_shuffle = train_set[0][shuffle_indexes]
            Y_shuffle = train_set[1][shuffle_indexes]
            for i in range(0, all_train_size, BATCH_SIZE):
                if i + BATCH_SIZE < all_train_size:
                    X_batch, Y_batch =  X_shuffle[i : i + BATCH_SIZE], Y_shuffle[i : i + BATCH_SIZE]
                    _, train_loss, train_acc, current_gradient = sess.run([self.train_op, self.loss, self.accuracy_train, self.gradient], {self.X:X_batch, self.Y:shift_label(Y_batch)})
                else:
                    X_batch, Y_batch =  X_shuffle[i:], Y_shuffle[i:]
                    _, train_loss, current_gradient = sess.run([self.train_op, self.loss, self.gradient], {self.X:X_batch, self.Y:shift_label(Y_batch)})
                    train_acc = -1
                step += 1
                if (np.sum(np.abs(current_gradient[0][0])) == 0 and np.sum(np.abs(current_gradient[1][0])) == 0) or ((self.use_crossentropy and train_loss <= CROSSENTROPY_THRESHOLD) or (not self.use_crossentropy and train_loss <= 0)):
                    minimum_point_counter += 1 
                if step % PRINT_STEP_JUMP == 0:
                    print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc)) 
            if np.ceil(all_train_size / BATCH_SIZE) == minimum_point_counter:
                print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc)) 
                break
            self.W, self.B = sess.run([self.W_tf, self.B_tf])
        print("NN Train accuracy: {0}".format(train_acc)) 
        return train_loss        

    def run(self, train_set, test_set, result_object=None, just_test=False):
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
            if self.W is not None:
                self.W_tf = tf.get_variable('W', initializer=self.W.T)
            else:
                self.W_tf = tf.get_variable('W', shape=[D, self.r])
            self.B_tf = tf.get_variable('B_W', initializer=self.B)
            if self.without_B0:
                self.B0_const = tf.constant(-1 * np.zeros([1], dtype=TYPE), dtype=TYPE)
            else:
                self.B0_tf = tf.get_variable('B_0', initializer=self.B0)
            
            # Netrowk
            out_1 = tf.nn.relu(tf.matmul(self.X, self.W_tf) + self.B_tf)
            if self.without_B0:
                logits = tf.reduce_sum(out_1, axis=1) + self.B0_const
            else:
                logits = tf.reduce_sum(out_1, axis=1)  + self.B0_tf

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
            if self.use_crossentropy:
                self.loss = tf.keras.losses.binary_crossentropy(self.Y, logits, from_logits=True)
            else:
                loss_vec = tf.losses.hinge_loss(logits=logits, labels=self.Y, reduction=tf.losses.Reduction.NONE)
                self.loss = tf.reduce_mean(loss_vec)
            
            # set optimizer
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            #optimizer = tf.train.AdamOptimizer(self.lr)
            self.gradient = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(self.gradient, global_step=global_step)

            with tf.Session() as sess:
                
                # init params
                init = tf.initialize_all_variables()
                sess.run(init)
                if not just_test:
                    if self.use_batch:
                        train_loss = self.train_with_batch(train_set, sess, result_object)    
                    else:
                        train_loss = self.train_without_batch(train_set, sess, result_object)

                test_loss, test_acc = sess.run([self.loss, accuracy_test], {self.X:test_set[0], self.Y:shift_label(test_set[1])})  
                print('NN Test accuracy: {0}'.format(test_acc)) 

                if self.without_B0:
                    self.W, self.B = sess.run([self.W_tf, self.B_tf])
                else:
                    self.W, self.B, self.B0 = sess.run([self.W_tf, self.B_tf, self.B0_tf])
                self.W = self.W.T

            return train_loss, test_acc