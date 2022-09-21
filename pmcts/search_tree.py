from math import log, sqrt
import random as pr

import numpy as np
from rdkit import Chem

from pmcts.utils import chem_kn_simulation, build_smiles_from_tokens, expanded_node, has_passed_through_filters


class Tree_Node():
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
        self.reward_calculator = reward_calculator
        self.val = conf['token']
        self.max_len=conf['max_len']

    def selection(self):
        ucb = []
        for i in range(len(self.childNodes)):
            ucb.append((self.childNodes[i].wins +
                        self.childNodes[i].virtual_loss) /
                       (self.childNodes[i].visits +
                        self.childNodes[i].num_thread_visited) +
                       self.conf['c_val'] *
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
        node_idxs = expanded_node(model, self.state, self.val)
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
        filter_flag = 0

        all_posible = chem_kn_simulation(chem_model, state, self.val, self.conf)
        smi = build_smiles_from_tokens(all_posible, self.val)
        if has_passed_through_filters(smi, self.conf):
            mol = Chem.MolFromSmiles(smi)
            values_list = [f(mol) for f in self.reward_calculator.get_objective_functions(self.conf)]
            score = self.reward_calculator.calc_reward_from_objective_values(values=values_list, conf=self.conf)
            filter_flag = 1
        else:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                values_list = [0 for f in self.reward_calculator.get_objective_functions(self.conf)]
            else:
                values_list = [f(mol) for f in self.reward_calculator.get_objective_functions(self.conf)]
            score = -1000 / (1 + 1000)
            filter_flag = 0
        return values_list, score, smi, filter_flag

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
