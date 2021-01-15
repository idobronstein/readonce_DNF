from consts import *

def calc_combinations(current, i, all_combinations, D):
    if i + 1 > D:
        all_combinations.append(current)
    else:
        current[i] = 1
        calc_combinations(current[:], i + 1, all_combinations, D)
        current[i] = -1
        calc_combinations(current[:], i + 1, all_combinations, D) 

def get_all_combinations(D=D):
    all_combinations = []
    init_state = [0] * D
    calc_combinations(init_state, 0, all_combinations, D)
    return all_combinations

def calc_partitions(current, s, all_partition):
    if s == 0:
        all_partition.append(current)
    for i in range(1, s + 1):
        current_new = current[:]
        current_new.append(i)
        calc_partitions(current_new, s - i, all_partition)

def get_all_partitions(D=D):
    all_partition = []
    calc_partitions([], D, all_partition)
    all_partition_order = []
    for partition in all_partition:
        partition.sort()
        if partition not in all_partition_order:
            all_partition_order.append(partition)
    return all_partition_order

def get_all_balanced_partitions(D=D):
    all_partition = get_all_partitions(D)
    is_balanced = lambda p: np.unique(p, return_counts=True)[1][0] == len(p)
    all_balanced_partitions = [partition for partition in all_partition if is_balanced(partition)]
    return all_balanced_partitions

def generate_samples(amount, D=D):
    return 2 * (np.random.randn(amount, D) > 0).astype(TYPE) - 1

def get_random_init_uniform_samples(set_size, D=D):
    x = (np.random.randn(set_size, D) > 0.0).astype(TYPE)
    x = 2 * (x-0.5)
    return x

def get_random_init_non_uniform_samples(set_size, positive_prob, D=D):
    return np.random.choice([POSITIVE, NEGATIVE], [set_size, D], p=[positive_prob, 1 - positive_prob])

class ReadOnceDNF():

    def __init__(self, partition=[], specifiec_DNF=None):
        if specifiec_DNF:
            self.DNF = specifiec_DNF
        else:
            self.DNF = []
            for i in range(len(partition)):
                term = []
                for j in range(len(partition)):
                    if i == j:
                        term += [1] * partition[j]
                    else:
                        term += [0] * partition[j]
                self.DNF.append(np.array(term, dtype=TYPE))

    def get_label(self, x):
        for term in self.DNF:
            flag = True
            for i, literal in enumerate(term):
                if literal * x[i] == NEGATIVE:
                    flag = False
            if flag:
                return POSITIVE
        return NEGATIVE

    def evaluate(self, X, Y):
        res = 0
        for (x, y) in zip(X, Y):
            if self.get_label(x) == y:
                res += 1
        return res / X.shape[0]