from consts import *
from data import *
from utilits import *


class NTKNetwork():

    def __init__(self, epsilon_init, lr, r=0, W_init=None, B_init=None, use_batch=False):
        print("Initialize fix layer two network with gaussin initialization")
        self.r = r 
        self.W = np.array(SIGMA * np.random.randn(self.r, D), dtype=TYPE)
        self.B = np.array(SIGMA * np.random.randn(self.r), dtype=TYPE)
        self.lr = lr
        self.all_W = [self.W]
        self.all_B = [self.B]
        self.use_batch = use_batch

    def train_without_batch(self, train_set, sess, mask, result_object=None):
        X_positive = train_set[0][[i for i in range(train_set[0].shape[0]) if train_set[1][i] == POSITIVE]]
        Y_positive = np.ones([X_positive.shape[0]], dtype=TYPE)
        for step in range(0, MAX_STEPS):
            _, train_loss, train_acc, current_gradient = sess.run([self.train_op, self.loss, self.accuracy_train, self.gradient], {self.X:train_set[0], self.Y:shift_label(train_set[1]), self.mask_tf: mask})
            positive_loss = sess.run([ self.loss], {self.X:train_set[0], self.Y:shift_label(train_set[1]), self.mask_tf: mask})[0]
            if (np.sum(np.abs(current_gradient[0][0])) == 0 and np.sum(np.abs(current_gradient[1][0])) == 0) or ( positive_loss <= CROSSENTROPY_THRESHOLD):
                print('step: {0}, loss: {1}, accuracy: {2}, positive_loss: {3}'.format(step, train_loss, train_acc, positive_loss)) 
                break 
            if step % PRINT_STEP_JUMP == 0:
                print('step: {0}, loss: {1}, accuracy: {2}, positive_loss: {3}'.format(step, train_loss, train_acc, positive_loss)) 
        print("NN Train accuracy: {0}".format(train_acc)) 
        return train_loss

    def train_with_batch(self, train_set, sess, mask, result_object=None):
        all_train_size = train_set[0].shape[0]
        step = 0
        
        while step < MAX_STEPS:
            minimum_point_counter = 0
            for i in range(0, all_train_size, BATCH_SIZE):
                if i + BATCH_SIZE < all_train_size:
                    X_batch, Y_batch =  train_set[0][i : i + BATCH_SIZE], train_set[1][i : i + BATCH_SIZE]
                    _, train_loss, train_acc, current_gradient = sess.run([self.train_op, self.loss, self.accuracy_train, self.gradient], {self.X:X_batch, self.Y:shift_label(Y_batch), self.mask_tf: mask})
                else:
                    X_batch, Y_batch =  train_set[0][i:], train_set[1][i:]
                    _, train_loss, current_gradient = sess.run([self.train_op, self.loss, self.gradient], {self.X:X_batch, self.Y:shift_label(Y_batch), self.mask_tf: mask})
                    train_acc = -1
                step += 1
                if (np.sum(np.abs(current_gradient[0][0])) == 0 and np.sum(np.abs(current_gradient[1][0])) == 0) or (train_loss <= CROSSENTROPY_THRESHOLD):
                    print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc)) 
                    minimum_point_counter += 1 
                if step % PRINT_STEP_JUMP == 0:
                    print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc)) 
            if i == minimum_point_counter:
                break
            self.W, self.B = sess.run([self.W_tf, self.B_tf])
            if result_object:
                result_object.save_result_to_pickle('Network.pkl', (self.W, self.B))
        print("NTK NN Train accuracy: {0}".format(train_acc)) 
        return train_loss        

    def create_mask(self, W0, X):
        mask = np.matmul(X, W0.T)
        mask[mask>0] = 1
        mask[mask<0] = 0
        return mask

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
            self.mask_tf = tf.placeholder(TYPE, name='mask', shape=[None, self.r])
            self.W0_tf = tf.constant(self.W.T, name='W0')
            self.B0_tf = tf.constant(self.B, name='B0')
            self.W_tf = tf.get_variable('W', initializer=self.W.T)
            self.B_tf = tf.get_variable('B_W', initializer=self.B)
            
            
            # Netrowk
            out_initizlize_layer = tf.reduce_sum(tf.nn.relu(tf.matmul(self.X, self.W0_tf) + self.B0_tf), axis=1) - 1 
            out_train = tf.multiply(tf.matmul(self.X, self.W_tf) + self.B_tf, self.mask_tf)
            logits_train = out_initizlize_layer + tf.reduce_sum(out_train, axis=1)

            # calc accuracy
            prediction_train = tf.round(tf.nn.sigmoid(logits_train))
            ones_train = tf.constant(np.ones([train_size]), dtype=TYPE)
            zeros_train = tf.constant(np.zeros([train_size]), dtype=TYPE)
            correct_train = tf.where(tf.equal(prediction_train, self.Y), ones_train, zeros_train)
            self.accuracy_train = tf.reduce_mean(correct_train)
            
            prediction_test = tf.round(tf.nn.sigmoid(logits_train))
            ones_test = tf.constant(np.ones([test_size]), dtype=TYPE)
            zeros_test = tf.constant(np.zeros([test_size]), dtype=TYPE)
            correct_test = tf.where(tf.equal(prediction_test, self.Y), ones_test, zeros_test)
            accuracy_test = tf.reduce_mean(correct_test)

            # calc loss
            #loss_vec = tf.losses.hinge_loss(logits=logits_train, labels=self.Y, reduction=tf.losses.Reduction.NONE)
            #self.loss = tf.reduce_mean(loss_vec)
            self.loss = tf.keras.losses.binary_crossentropy(self.Y, logits_train, from_logits=True)
            
            # set optimizer
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.gradient = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(self.gradient, global_step=global_step)

            with tf.Session() as sess:
                # init params
                init = tf.initialize_all_variables()
                sess.run(init)
                
                # train
                train_mask = self.create_mask(self.W, train_set[0])
                if not just_test:
                    if self.use_batch:
                        train_loss = self.train_with_batch(train_set, sess, train_mask, result_object)    
                    else:
                        train_loss = self.train_without_batch(train_set, sess, train_mask, result_object)
                    
                test_acc = sess.run([accuracy_test], {self.X:test_set[0], self.Y:shift_label(test_set[1]), self.mask_tf: self.create_mask(self.W, test_set[0])})[0] 
                print('NTK NN Test accuracy: {0}'.format(test_acc)) 

                self.W, self.B = sess.run([self.W_tf, self.B_tf])
                self.W = self.W.T

            return train_loss, test_acc