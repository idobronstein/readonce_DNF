from consts import *

def calc_combinations(current, i, all_combinations):
    if i + 1 > D:
        all_combinations.append(current)
    else:
        current[i] = 1
        calc_combinations(current[:], i + 1, all_combinations)
        current[i] = -1
        calc_combinations(current[:], i + 1, all_combinations) 

def get_all_combinations():
    all_combinations = []
    init_state = [0] * D
    calc_combinations(init_state, 0, all_combinations)
    return all_combinations

def calc_partitions(current, s, all_partition):
    if s == 0:
        all_partition.append(current)
    for i in range(1, s + 1):
        current_new = current[:]
        current_new.append(i)
        calc_partitions(current_new, s - i, all_partition)

def get_all_partitions():
    all_partition = []
    calc_partitions([], D, all_partition)
    all_partition_order = []
    for partition in all_partition:
        partition.sort()
        if partition not in all_partition_order:
            all_partition_order.append(partition)
    return all_partition_order

def get_all_balanced_partitions():
    all_partition = get_all_partitions()
    is_balanced = lambda p: np.unique(p, return_counts=True)[1][0] == len(p)
    all_balanced_partitions = [partition for partition in all_partition if is_balanced(partition)]
    return all_balanced_partitions

def upsampling(X, Y, amount):
    print("Upsampeling with amount {0}".format(amount))
    for i in range(2 ** D):
            for _ in range(amount):
                X = np.concatenate([X, [X[i]]])
                Y = np.concatenate([Y, [Y[i]]])
    print("Number of samples {0}".format(X.shape[0]))
    return X, Y

def downsampling(X, Y, prob):
    print("Downsampeling with prob {0}".format(prob))
    for i in range(2 ** D - 1, -1 , -1):
            if np.random.uniform() > prob:
                X = np.delete(X, i, 0)
                Y = np.delete(Y, i, 0)
    print("Number of samples {0}".format(X.shape[0]))
    return X, Y

class ReadOnceDNF():

    def __init__(self, partition):
        self.DNF = []
        for i in range(len(partition)):
            term = []
            for j in range(len(partition)):
                if i == j:
                    term += [1] * partition[j]
                else:
                    term += [0] * partition[j]
            self.DNF.append(term)


    def get_label(self, x):
        for term in self.DNF:
            flag = True
            for i, literal in enumerate(term):
                if literal * x[i] == NEGATIVE:
                    flag = False
            if flag:
                return POSITIVE
        return NEGATIVE