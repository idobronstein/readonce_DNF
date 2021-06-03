from consts import *
from utilities import *

class ConvexNetwork():

    def __init__(self, lr, r, sigma=SIGMA):
        self.r = r
        self.lr = lr
        self.W = np.array(sigma * np.random.randn(self.r, D), dtype=TYPE)
        self.B = np.zeros([self.r], dtype=TYPE) 
        self.B0 = np.zeros([1], dtype=TYPE)

    def train(self, train_set, sess):
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
                if (np.sum(np.abs(current_gradient[0][0])) == 0 and np.sum(np.abs(current_gradient[1][0])) == 0) or (train_loss <= CROSSENTROPY_THRESHOLD):
                    minimum_point_counter += 1 
                if step % PRINT_STEP_JUMP == 0:
                    print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc)) 
            if np.ceil(all_train_size / BATCH_SIZE) == minimum_point_counter:
                print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc)) 
                break
            self.W, self.B = sess.run([self.W_tf, self.B_tf])
        print("Convex netwrok Train accuracy: {0}".format(train_acc)) 
        return train_loss        

    def run(self, train_set, test_set):
        train_size = BATCH_SIZE
        test_size = test_set[0].shape[0]
        train_loss, test_acc = 0, 0
        tf.reset_default_graph()
        with tf.Graph().as_default():
            # init variables
            global_step = tf.Variable(0, trainable=False, name='global_step')
            self.X = tf.placeholder(TYPE, name='X', shape=[None, D])
            self.Y = tf.placeholder(TYPE, name='Y', shape=[None])
            self.W_tf = tf.get_variable('W', initializer=self.W.T)
            self.B_tf = tf.get_variable('B_W', initializer=self.B)
            self.B0_tf = tf.get_variable('B_0', initializer=self.B0)
            
            # Netrowk
            out_1 = tf.nn.relu(tf.matmul(self.X, self.W_tf) + self.B_tf)
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
            self.loss = tf.keras.losses.binary_crossentropy(self.Y, logits, from_logits=True)
            
            # set optimizer
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.gradient = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(self.gradient, global_step=global_step)

            with tf.Session() as sess:
                
                # init params
                init = tf.initialize_all_variables()
                sess.run(init)
                
                # train
                train_loss = self.train(train_set, sess)    

                # test
                test_loss, test_acc = sess.run([self.loss, accuracy_test], {self.X:test_set[0], self.Y:shift_label(test_set[1])})  
                print('Convex netwrok Test accuracy: {0}'.format(test_acc)) 

                self.W, self.B, self.B0 = sess.run([self.W_tf, self.B_tf, self.B0_tf])
                self.W = self.W.T

            return train_loss, test_acc