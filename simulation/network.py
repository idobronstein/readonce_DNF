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

    def update_network(self, X, Y, lr, display):
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
                        step = step if y == POSITIVE else - lr / step
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
        if display:
            print("Accuracy: {0} / {1}".format(zero_loss_sample_counter, X.shape[0]))
        # Return if we are in a global minimum 
        return global_minimum_point, local_minimum_point
                          

##################################### OLD #####################################


class TwoVariableNetwork():

    def __init__(self, a, b, L):
        self.a = a
        self.b = b
        self.L = L
        self.all_a = [a]
        self.all_b = [b]
        self.all_gradient = []

    def get_neuron_value(self, x, i):
        if i == 0:
            w = np.array([self.a] * self.L + [OPISITE_VALUE - self.a] * self.L, dtype=TYPE)
        else:
            w = np.array([OPISITE_VALUE - self.a] * self.L + [self.a] * self.L, dtype=TYPE)
        value = np.dot(x, w) + self.b
        return value if value > ZERO_THRESHOLD else 0

    def get_network_value(self, x):
        value = 0
        for i in range(2):
            a = self.get_neuron_value(x, i)
            value += a if a > 0 else 0
        return value

    def check_loss_reset_sample(self, x, y):
        value = self.get_network_value(x)
        return (value >= MAX_VALUE_FOR_POSITIVE_SAMPLE and y == POSITIVE) or (value <= MIN_VALUE_FOR_NEGATIVE_SAMPLE and y == NEGATIVE)

    def check_relu_reset_sample(self, x, i):
        return self.get_neuron_value(x, i) <= 0

    def update_network(self, X, Y, if_print=False):
        minimum_point = True
        gradient_step_a, gradient_step_b = 0, 0

        # Update all weights
        for x, y in zip(X, Y):
            if not self.check_loss_reset_sample(x, y):
                minimum_point = False
                if not self.check_relu_reset_sample(x, 0):
                    v = np.sum(x[:self.L]) - np.sum(x[self.L:])
                    if if_print:
                        print(0,y , x,v)
                    gradient_step_a += y * v / X.shape[0]
                    gradient_step_b += y / X.shape[0]
                if not self.check_relu_reset_sample(x, 1):
                    v = - np.sum(x[:self.L]) + np.sum(x[self.L:])
                    #if if_print:
                    #    print(1,y,x,v)
                    gradient_step_a += y * v / X.shape[0]
                    gradient_step_b += y / X.shape[0]
        # Save new weights
        self.a = self.a + LR * gradient_step_a
        self.b = self.b+ LR * gradient_step_b
        self.all_a.append(self.a)
        self.all_b.append(self.b)
        self.all_gradient.append((gradient_step_a, gradient_step_b))
        # Return if we are in a global minimum 
        return minimum_point
