from consts import *

class Network():

    def __init__(self, W, B):
        assert W.shape[0] == B.shape[0]
        assert W.shape[1] == D  
        self.W = W
        self.B = B
        self.r = W.shape[0]
        self.all_W = [W]
        self.all_B = [B]

    def get_value(self, x, w, b):
        value = 0
        for j in range(D):
            value += w[j] if x[j] == POSITIVE else - w[j]
        value += b
        return value if value > 0 else 0

    def get_neuron_value(self, x, i):
        return self.get_value(x, self.W[i], self.B[i])

    def get_past_neuron_value(self, x, i, step):
        return self.get_value(x, self.all_W[step][i], self.all_B[step][i])

    def get_past_network_value(self, x, step):
        value = 0
        for i in range(self.r):
            a = self.get_past_neuron_value(x, i, step)
            value += a
        return value

    def get_network_value(self, x):
        value = 0
        for i in range(self.r):
            a = self.get_neuron_value(x, i)
            value += a
        return value

    def get_prderiction(self, x):
        first_layer_value = self.get_network_value(x)
        return np.sign(first_layer_value + SECOND_LAYER_BAIS)

    def check_loss_reset_sample(self, x, y):
        value = self.get_network_value(x)
        return (value >= MAX_VALUE_FOR_POSITIVE_SAMPLE and y == POSITIVE) or (value <= MIN_VALUE_FOR_NEGATIVE_SAMPLE and y == NEGATIVE)

    def check_relu_reset_sample(self, x, i):
        return self.get_neuron_value(x, i) <= 0

    def check_predriction_for_dataset(self, sess):
        current_margin = sess.run([self.mask_two_margin], {self.tf_W: self.W, self.tf_B: self.B})
        return np.sum(current_margin)

    def prepere_update_network(self, X, Y):
        self.tf_W = tf.placeholder(TYPE, name='W', shape=[self.r, D])
        self.tf_B = tf.placeholder(TYPE, name='B', shape=[self.r])
        tf_X = tf.constant(X, dtype=TF_TYPE, name="X")
        tf_Y = tf.constant(Y, dtype=TF_TYPE, name="Y")
        # Create matrix
        self.X_matrix = tf.concat([tf_X, tf.ones([X.shape[0],1], dtype=TF_TYPE)], axis=1)
        self.W_matrix = tf.concat([tf.transpose(self.tf_W), [self.tf_B]], axis=0)
        # Calc first layer
        self.first_layer_output = tf.matmul(self.X_matrix, self.W_matrix)
        self.mask_first_layer_output= tf.where(self.first_layer_output < 0, tf.zeros(self.first_layer_output.shape, dtype=TF_TYPE), self.first_layer_output)
        # Calc second layer
        self.second_layer_output = tf.reshape(tf.matmul(self.mask_first_layer_output, tf.ones([self.r,1], dtype=TF_TYPE)), [X.shape[0]]) + SECOND_LAYER_BAIS
        # Calc margin
        self.margin = tf.multiply(tf.transpose(self.second_layer_output), tf_Y)
        self.mask_one_margin = tf.where(self.margin < 0, tf.zeros(self.margin.shape, dtype=TF_TYPE), self.margin)
        self.mask_two_margin = tf.where(self.margin > 0, tf.ones(self.margin.shape, dtype=TF_TYPE), self.mask_one_margin)
        # Calc reset hinge loss map
        self.hinge_loss_map = HINGE_LOST_CONST - self.margin
        self.mask_one_hinge_loss_map = tf.where(self.hinge_loss_map < 0, tf.zeros(self.hinge_loss_map.shape, dtype=TF_TYPE), self.hinge_loss_map)
        self.mask_two_hinge_loss_map = tf.where(self.hinge_loss_map > 0, tf.ones(self.hinge_loss_map.shape, dtype=TF_TYPE), self.mask_one_hinge_loss_map)
        # Calc relu reset map
        self.relu_reset_map = tf.transpose(self.mask_first_layer_output)
        self.mask_relu_reset_map= tf.where(self.relu_reset_map > 0, tf.ones(self.relu_reset_map.shape, dtype=TF_TYPE), self.relu_reset_map)
        # Calc update rule
        self.total_reset_map = tf.multiply(self.mask_relu_reset_map, self.mask_two_hinge_loss_map)
        self.total_map = tf.multiply(self.total_reset_map, tf_Y)
        self.w_update_rule = tf.matmul(self.total_map, tf_X)
        self.b_update_rule = tf.reduce_sum(self.total_map, axis=1)
              
    def update_network(self, sess, lr):
        global_minimum_point, local_minimum_point = True, True
        current_w_update_rule, current_b_update_rule, current_hinge_loss_map = sess.run([self.w_update_rule, self.b_update_rule, self.mask_two_hinge_loss_map], {self.tf_W: self.W, self.tf_B: self.B})
        
        # Update the weights
        self.W = self.W + lr*current_w_update_rule
        self.B = self.B + lr*current_b_update_rule
        #self.all_W.append(self.W)
        #self.all_B.append(self.B)
        # Check if we are in local minimum point
        local_minimum_point = np.sum(np.abs(current_w_update_rule)) == 0 and np.sum(np.abs(current_b_update_rule)) == 0
        # Check if we are in global minimum point
        non_zero_loss_sample_counter = np.sum(current_hinge_loss_map)
        global_minimum_point = non_zero_loss_sample_counter == 0
        # Return if we are in a global minimum 
        return global_minimum_point, local_minimum_point, non_zero_loss_sample_counter

##################################### OLD #####################################

    def update_network_old(self, X, Y, lr, result_objuct=None):
        # Update all weights
        global_minimum_point, local_minimum_point = True, True
        gradient_step_W, gradient_step_B = np.zeros([self.r, D]), np.zeros([self.r])
        zero_loss_sample_counter = 0
        for x, y in zip(X, Y):
            if not self.check_loss_reset_sample(x, y):
                global_minimum_point = False
                for i in range(self.r): 
                    if not self.check_relu_reset_sample(x, i):
                        step = int(lr / X.shape[0])
                        step = step if y == POSITIVE else - step
                        for j in range(D):
                             gradient_step_W[i][j] += step if x[j] == POSITIVE else -step     
                        gradient_step_B[i] += step
            else:
                zero_loss_sample_counter += 1
        # Save new weights
        self.W = self.W + gradient_step_W
        self.B = self.B + gradient_step_B
        self.all_W.append(self.W)
        self.all_B.append(self.B)
        # Check if we are in local minimum point
        if np.sum(np.abs(gradient_step_W)) > 0 or np.sum(np.abs(gradient_step_B)) > 0:
            local_minimum_point = False
        # Display accuracy
        if result_objuct:
            result_object.rootLogger.critical("Accuracy: {0} / {1}".format(zero_loss_sample_counter, X.shape[0]))
        # Return if we are in a global minimum 
        return global_minimum_point, local_minimum_point