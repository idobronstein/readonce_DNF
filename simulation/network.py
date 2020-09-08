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

    def check_predriction_for_dataset(self, X, Y):
        predriction_count = 0
        for x, y in zip(X, Y):
            if self.get_prderiction(x) == y:
                predriction_count += 1
        return predriction_count
                          
    def update_network(self, X, Y, lr):
        global_minimum_point, local_minimum_point = True, True
        # Create matrix
        X_matrix = np.concatenate([X, np.ones([X.shape[0],1], dtype=TYPE)], axis=1)
        W_matrix = np.concatenate([self.W.T, [self.B]])
        # Calc first layer
        first_layer_output = np.matmul(X_matrix, W_matrix)
        first_layer_output[first_layer_output < 0] = 0
        # Calc second layer
        second_layer_output = np.matmul(first_layer_output, np.ones([self.r], dtype=TYPE)) + SECOND_LAYER_BAIS
        # Calc reset hinge loss map
        hinge_loss_map = HINGE_LOST_CONST - np.multiply(second_layer_output.T, Y)
        hinge_loss_map[hinge_loss_map < 0] = 0
        hinge_loss_map[hinge_loss_map > 0] = 1
        # Calc relu reset map
        relu_reset_map = first_layer_output.T
        relu_reset_map[relu_reset_map > 0] = 1
        # Calc update rule
        total_reset_map = np.multiply(relu_reset_map, hinge_loss_map)
        total_map = np.multiply(total_reset_map, Y)
        w_update_rule = np.matmul(total_map, X)
        b_update_rule = np.sum(total_map, axis=1)
        # Update the weights
        self.W = self.W + lr*w_update_rule
        self.B = self.B + lr*b_update_rule
        self.all_W.append(self.W)
        self.all_B.append(self.B)
        # Check if we are in local minimum point
        local_minimum_point = np.sum(np.abs(w_update_rule)) == 0 and np.sum(np.abs(b_update_rule)) == 0
        # Check if we are in global minimum point
        non_zero_loss_sample_counter = np.sum(hinge_loss_map)
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