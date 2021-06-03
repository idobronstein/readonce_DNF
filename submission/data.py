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

def generate_samples(amount, D=D):
    return 2 * (np.random.randn(amount, D) > 0).astype(TYPE) - 1

def get_random_init_uniform_samples(set_size, D=D):
    x = (np.random.randn(set_size, D) > 0.0).astype(TYPE)
    x = 2 * (x-0.5)
    return x

class ReadOnceDNF():

    def __init__(self, partition=[], specifiec_DNF=None):
        if specifiec_DNF is not None:
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