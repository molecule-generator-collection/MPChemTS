
from keras.preprocessing import sequence
import gzip
import numpy as np
from math import log, sqrt
import random as pr

from pmcts.property_simulator import simulator
from pmcts import sascorer
from pmcts.rollout import chem_kn_simulation, chem_kn_simulation_stateful, predict_smile, make_input_smile

from mpi4py import MPI

class Tree_Node(simulator):
    """
    define the node in the tree
    """
    def __init__(self, state, parentNode=None, property=property):
        # todo: these should be in a numpy array
        # MPI payload [node.state, node.reward, node.wins, node.visits, node.num_thread_visited, node.path_ucb]
        self.state = state
        self.reward = 0
        self.wins = 0
        self.visits = 0
        self.num_thread_visited = 0
        self.path_ucb = []

        self.virtual_loss = 0
        self.childNodes = []
        self.parentNode = parentNode
        self.check_childnode = []
        self.expanded_nodes = []
        self.childucb = []

        # todo: move simulator outside of Tree_Node
        self.property=property
        simulator.__init__(self, property)

    def selection(self):
        ucb = []
        for i in range(len(self.childNodes)):
            ucb.append((self.childNodes[i].wins +
                        self.childNodes[i].virtual_loss) /
                       (self.childNodes[i].visits +
                        self.childNodes[i].num_thread_visited) +
                       1.0 *
                       sqrt(2 *log(self.visits +self.num_thread_visited) /
                            (self.childNodes[i].visits +
                                self.childNodes[i].num_thread_visited)))
        self.childucb = ucb
        m = np.amax(ucb)
        indices = np.nonzero(ucb == m)[0]
        ind = pr.choice(indices)
        self.childNodes[ind].num_thread_visited += 1
        self.num_thread_visited += 1
        return ind, self.childNodes[ind]

    def expansion(self, model):
        state = self.state
        all_nodes = []
        end = "\n"
        position = []
        position.extend(state)
        total_generated = []
        new_compound = []
        get_int_old = []
        for j in range(len(position)):
            get_int_old.append(self.val.index(position[j]))
        get_int = get_int_old
        x = np.reshape(get_int, (1, len(get_int)))
        x_pad = x
#        x_pad = sequence.pad_sequences(x, maxlen=self.max_len, dtype='int32',
#                                       padding='post', truncating='pre', value=0.)
        model.reset_states()
        predictions = model.predict(x_pad)
        preds = np.asarray(predictions[0]).astype('float64')
    #    preds = np.asarray(predictions[0][len(get_int) - 1]).astype('float64')
        preds = np.log(preds) / 1.0
        preds = np.exp(preds) / np.sum(np.exp(preds))
        sort_index = np.argsort(-preds)
        i = 0
        sum_preds = preds[sort_index[i]]
        all_nodes.append(sort_index[i])
        while sum_preds <= 0.95:
            i += 1
            all_nodes.append(sort_index[i])
            sum_preds += preds[sort_index[i]]
        self.check_childnode.extend(all_nodes)
        self.expanded_nodes.extend(all_nodes)

    def addnode(self, m):
        self.expanded_nodes.remove(m)
        added_nodes = []
        added_nodes.extend(self.state)
        added_nodes.append(self.val[m])
        self.num_thread_visited += 1
        n = Tree_Node(state=added_nodes, parentNode=self, property=self.property)
        n.num_thread_visited += 1
        self.childNodes.append(n)
        return n

    def update_local_node(self, score):
        self.visits += 1
        self.wins += score
        self.reward = score

    def simulation(self, chem_model, state, rank, gauid):
#        all_posible = chem_kn_simulation(chem_model, state, self.val, self.max_len)
        all_posible = chem_kn_simulation_stateful(chem_model, state, self.val, self.max_len)
        generate_smile = predict_smile(all_posible, self.val)
        new_compound = make_input_smile(generate_smile)
        score, mol= self.run_simulator(new_compound,rank)
        return score, mol

    def backpropagation(self, cnode):
        self.wins += cnode.reward
        self.visits += 1
        self.num_thread_visited -= 1
        self.reward = cnode.reward
        for i in range(len(self.childNodes)):
            if cnode.state[-1] == self.childNodes[i].state[-1]:
                self.childNodes[i].wins += cnode.reward
                self.childNodes[i].num_thread_visited -= 1
                self.childNodes[i].visits += 1
