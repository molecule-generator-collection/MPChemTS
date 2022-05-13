import itertools

import numpy as np
from math import log, sqrt
import random as pr

#from mpi4py import MPI

from pmcts.property_simulator import simulator
from pmcts.utils import chem_kn_simulation, build_smiles_from_tokens

class Tree_Node(simulator):
    """
    define the node in the tree
    """
    def __init__(self, state, parentNode=None, reward_calculator=None, conf=None):
        # todo: these should be in a numpy array
        # MPI payload [node.state, node.reward, node.wins, node.visits, node.num_thread_visited, node.path_ucb]
        self.state = state
        self.childNodes = []
        self.parentNode = parentNode
        self.wins = 0
        self.visits = 0
        self.virtual_loss = 0
        self.num_thread_visited = 0
        self.reward = 0
        self.check_childnode = []
        self.expanded_nodes = []
        self.path_ucb = []
        self.childucb = []
        self.conf = conf
        simulator.__init__(self, reward_calculator, self.conf)
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

    def expansion(self, model, threshold=0.95):
        state = self.state
        get_int = [self.val.index(state[j]) for j in range(len(state))]
        x = np.reshape(get_int, (1, len(get_int)))
        model.reset_states()
        preds = model.predict(x)
        state_preds = np.squeeze(preds)
        sorted_idxs = np.argsort(state_preds)[::-1]
        sorted_preds = state_preds[sorted_idxs]
        for i, v in enumerate(itertools.accumulate(sorted_preds)):
            if v > threshold:
                i = i if i != 0 else 1  # return one index if the first prediction value exceeds the threshold.
                break 
        node_idxs = sorted_idxs[:i]
        self.check_childnode.extend(node_idxs)
        self.expanded_nodes.extend(node_idxs)

    def addnode(self, m):
        self.expanded_nodes.remove(m)
        added_nodes = []
        added_nodes.extend(self.state)
        added_nodes.append(self.val[m])
        self.num_thread_visited += 1
        n = Tree_Node(state=added_nodes, parentNode=self, conf=self.conf)
        n.num_thread_visited += 1
        self.childNodes.append(n)
        return n

    def update_local_node(self, score):
        self.visits += 1
        self.wins += score
        self.reward = score

    def simulation(self, chem_model, state, rank, gauid):
        all_posible = chem_kn_simulation(chem_model, state, self.val, self.conf)
        new_compound = build_smiles_from_tokens(all_posible, self.val)
        score, mol= self.run_simulator(new_compound, self.conf)
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
