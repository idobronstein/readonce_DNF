import sys
import os
sys.path.insert(-1, os.path.join(os.getcwd(), '..', 'common'))

from consts import *
from data import *
from result import *
from fix_layer_2_netowrk import *
from two_layer_network import *
from NTK_svn import *
from NTK_network import *
from mariano import *


def cluster_graph_w(network, add_to_name=''):
    leaves_index = cluster_network(network)

    # Plot W graph
    fig = pylab.figure()
    fig.suptitle('f2 - Cluster W', fontsize=20, x=0.6)
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    weights_by_leaves = network.W[leaves_index,:]
    im = axmatrix.matshow(weights_by_leaves, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
    pylab.colorbar(im, cax=axcolor)
    fig.savefig(os.path.join(r"~/temp", add_to_name + "cluser.png"), bbox_inches="tight")
    plt.close(fig)

def main():
    result_path = TEMP_RESULT_PATH if IS_TEMP else GENERAL_RESULT_PATH
    print("Making result object in the path: {0}".format(result_path))
    result_object = Result(result_path, IS_TEMP, extra_to_name='plot')

    print("Start a run for: {0}".format(DNF))        
    run_name = '_'.join([str(i) for i in DNF]) 
    result_object.create_dir(run_name)

    all_readonce = [
                        ReadOnceDNF(specifiec_DNF=[[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],[0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],[0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0]]),
                        ReadOnceDNF(specifiec_DNF=[[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],[0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],[0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0]]),
                        ReadOnceDNF(specifiec_DNF=[[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],[0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],[0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0]]),
                        ReadOnceDNF(specifiec_DNF=[[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],[0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],[0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0]])
                    ]       

    for round_num in range(NUM_OF_RUNNING):
        #X = np.array(get_all_combinations(), dtype=TYPE)
        X = get_random_init_uniform_samples(TRAIN_SIZE, D)
        X_test = get_random_init_uniform_samples(TEST_SIZE, D)
        for i, readonce in enumerate(all_readonce):
            Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)
            Y_test = np.array([readonce.get_label(x) for x in X_test], dtype=TYPE)
            train_set = (X, Y)
            test_set = (X_test, Y_test)

            network = FixLayerTwoNetwork(False, LR, R)
            network.run(train_set, test_set)
            #network = NTKNetwork(False, LR, R)
            #network.run(train_set, test_set)
            #result_object.cluster_graph(network, str(round_num))
            #network = FixLayerTwoNetwork(False, LR, R)
            #network.run(train_set, test_set)
            #network = mariano()
            #network.run(train_set, test_set)
            cluster_graph_w(network, add_to_name=str(i))
            #   import IPython; IPython.embed()     
main() 
