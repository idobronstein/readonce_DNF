from consts import *
from data import *
from result import *
from network import *

def clac_two_variable_loss(a, b, X, Y, L):
    w1 = np.array([a] * L + [OPISITE_VALUE - a] * L, dtype=TYPE)
    w2 = np.array([OPISITE_VALUE - a] * L + [a] * L, dtype=TYPE)
    total_loss = 0
    for x, y in zip(X, Y):
        N1 = max([0, np.dot(x, w1) + b]) + max([0, np.dot(x, w2) + b])
        N2 = N1 - 1
        sample_loss = max([0, 1 - y * N2])
        total_loss += sample_loss
    return total_loss

def main():
    print("Making result object in the path: {0}".format(GENERAL_RESULT_PATH))
    result_object = Result(False)
    print("Generate all partitions")
    all_partitions = get_all_partitions()
    all_partitions.remove([D])
    print("Generate all combinations")
    all_combinations = get_all_combinations()
    r = len(all_combinations)

    print("Create 2 terms DNF")
    assert D % 2 == 0, "D has to be even"
    L = int(D/2)
    read_once_DNF = ReadOnceDNF([L, L])

    print("Create Data")
    X = np.array(all_combinations, dtype=TYPE)
    Y = np.array([read_once_DNF.get_label(x) for x in X], dtype=TYPE)

    print("Calculate loss surface")
    a_grid, b_grid = np.meshgrid(A_RANGE, B_RANGE)
    loss_surface = np.zeros(a_grid.shape)
    for i in range(a_grid.shape[0]):
        for j in range(a_grid.shape[1]):
            loss_surface[i][j] = clac_two_variable_loss(a_grid[i][j], b_grid[i][j], X, Y, L)

    print("Intialize netwrok with: a={0}, b={1}".format(A_INIT, B_INIT))
    network = TwoVariableNetwork(A_INIT, B_INIT, L)

    print("Start running the network")
    step = 0
    minimum_point = False
    while not minimum_point and step < MAX_STEPS:
        if step % PRINT_STEP_JUMP == 0 and step > 0:
            print("Step number: {0}".format(step))
        #import ipdb; ipdb.set_trace()
        minimum_point = network.update_network(X, Y)
        step += 1

    gradient_path = [clac_two_variable_loss(network.all_a[i], network.all_b[i], X, Y, L) for i in range(step + 1)]

    result_object.show_3D_graph(a_grid, b_grid, loss_surface, network, gradient_path)

    return network, gradient_path

network, gradient_path = main()
