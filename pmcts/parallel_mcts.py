#from math import *
from asyncore import file_wrapper
from curses import raw
import time
import random
import numpy as np
from copy import deepcopy
from mpi4py import MPI
from collections import deque
from pmcts.check_ucbpath import backtrack_tdsdfuct, backtrack_mpmcts, compare_ucb_tdsdfuct, compare_ucb_mpmcts, update_selection_ucbtable_mpmcts, update_selection_ucbtable_tdsdfuct
from pmcts.search_tree import Tree_Node
from pmcts.zobrist_hash import Item, HashTable
from enum import Enum
import pandas as pd
import os

"""
classes defined distributed parallel mcts
"""


class JobType(Enum):
    '''
    defines JobType tag values
    values higher than PRIORITY_BORDER (128) mean high prority tags
    FINISH is not used in this implementation. It will be needed for games.
    '''
    SEARCH = 0
    BACKPROPAGATION = 1
    PRIORITY_BORDER = 128
    GATHER_RESULTS = 253
    TIMEUP = 254
    FINISH = 255

    @classmethod
    def is_high_priority(self, tag):
        return tag >= self.PRIORITY_BORDER.value


class p_mcts:
    """
    parallel mcts algorithms includes TDS-UCT, TDS-df-UCT and MP-MCTS
    """
    # todo: use generated_dict

    def __init__(self, communicator, chem_model, reward_calculator, conf):
        self.comm = communicator
        self.rank = self.comm.Get_rank()
        self.nprocs = self.comm.Get_size()

        self.chem_model = chem_model
        self.reward_calculator = reward_calculator
        self.conf = conf
        # Initialize HashTable
        root_node = Tree_Node(state=['&'], reward_calculator=reward_calculator, conf=conf)
        random.seed(3)
        self.hsm = HashTable(self.nprocs, root_node.val, root_node.max_len, len(root_node.val))

        if self.rank == 0:
            self.output_path = os.path.join(conf['output_dir'], f"result_C{conf['c_val']}.csv")
            if os.path.exists(self.output_path) and not conf['restart']:
                sys.exit(f"[ERROR] {self.output_path} already exists. Please specify a different file name.")
                assert(False)

        self.start_time = time.time()

        self.id_suffix = "_" + str(self.rank).zfill(len(str(self.nprocs)))

        # for results
        self.total_valid_num = 0

        # these are gathred to rank 0 in the end
        self.generated_id_list = []
        self.valid_smiles_list = []
        self.depth_list = []
        self.objective_values_list = [] # raw reward (could be list)
        self.reward_values_list = [] # normalized reward for UCT
        self.elapsed_time_list = []

        self.generated_dict = {}  # dictionary of generated compounds
        self.filter_check_list = [] # only needed for output (at rank 0)
        self.obj_column_names = [f.__name__ for f in self.reward_calculator.get_objective_functions(self.conf)]

    def get_generated_id(self):
        id = str(self.total_valid_num) + self.id_suffix
        self.total_valid_num += 1
        return id

    def elapsed_time(self):
        return time.time() - self.start_time

    def send_message(self, node, dest, tag, data=None):
        # send node using MPI_Bsend
        # typical usage of data is path_ucb for newly created child nodes
        if data is None:
            self.comm.bsend(np.asarray(
                [node.state, node.reward, node.wins, node.visits, node.num_thread_visited, node.path_ucb]),
                dest=dest, tag=tag)
        else:
            self.comm.bsend(np.asarray(
                [node.state, node.reward, node.wins, node.visits, node.num_thread_visited, data]),
                dest=dest, tag=tag)

    def send_search_childnode(self, node, ucb_table, dest):
        self.comm.bsend(np.asarray(
            [node.state, node.reward, node.wins, node.visits, node.num_thread_visited, ucb_table]),
            dest=dest, tag=JobType.SEARCH.value)

    def send_backprop(self, node, dest):
        self.comm.bsend(np.asarray(
            [node.state, node.reward, node.wins, node.visits, node.num_thread_visited, node.path_ucb]),
            dest=dest, tag=JobType.BACKPROPAGATION.value)

    def record_result(self, smiles, depth, reward, raw_reward_list, filter_flag):
        self.valid_smiles_list.append(smiles)
        self.depth_list.append(depth)
        self.reward_values_list.append(reward)
        self.elapsed_time_list.append(self.elapsed_time())
        self.generated_id_list.append(self.get_generated_id())
        self.objective_values_list.append(raw_reward_list)
        self.filter_check_list.append(filter_flag)

    def gather_results(self):
        status = MPI.Status()
        for id in range(1, self.nprocs):
            if self.rank == 0:
                (valid_smiles_list, depth_list, reward_values_list, elapsed_time_list,
                 generated_id_list, objective_values_list, filter_check_list) = self.comm.recv(source=id, tag=JobType.GATHER_RESULTS.value, status=status)
                self.valid_smiles_list.extend(valid_smiles_list)
                self.depth_list.extend(depth_list)
                self.reward_values_list.extend(reward_values_list)
                self.elapsed_time_list.extend(elapsed_time_list)
                self.generated_id_list.extend(generated_id_list)
                self.objective_values_list.extend(objective_values_list)
                self.filter_check_list.extend(filter_check_list)
            else:
                self.comm.send((self.valid_smiles_list, self.depth_list, self.reward_values_list, self.elapsed_time_list,
                                self.generated_id_list, self.objective_values_list, self.filter_check_list),
                                dest=0, tag=JobType.GATHER_RESULTS.value)

    def flush(self):
        df = pd.DataFrame({
            "generated_id": self.generated_id_list,
            "smiles": self.valid_smiles_list,
            "reward": self.reward_values_list,
            "depth": self.depth_list,
            "elapsed_time": self.elapsed_time_list,
            "is_through_filter": self.filter_check_list,
        })
        df_obj = pd.DataFrame(self.objective_values_list, columns=self.obj_column_names)
        df = pd.concat([df, df_obj], axis=1)
        if os.path.exists(self.output_path):
            df.to_csv(self.output_path, mode='a', index=False, header=False)
        else:
            df.to_csv(self.output_path, mode='w', index=False)
        # self.logger.info(f"save results at {self.output_path}")

        self.generated_id_list.clear()
        self.valid_smiles_list.clear()
        self.reward_values_list.clear()
        self.depth_list.clear()
        self.elapsed_time_list.clear()
        self.filter_check_list.clear()
        self.objective_values_list.clear()

    def TDS_UCT(self):
        # self.comm.barrier()
        status = MPI.Status()
        gau_id = 0  # this is used for wavelength

        self.start_time = time.time()
        _, rootdest = self.hsm.hashing(['&'])
        jobq = deque()
        timeup = False
        if self.rank == rootdest:
            root_job_message = np.asarray([['&'], None, 0, 0, 0, []])
            for i in range(3 * self.nprocs):
                temp = deepcopy(root_job_message)
                root_job = (JobType.SEARCH.value, temp)
                jobq.appendleft(root_job)
        while not timeup:
            if self.rank == 0:
                if self.elapsed_time() > 600:
                    timeup = True
                    for dest in range(1, self.nprocs):
                        dummy_data = tag = JobType.TIMEUP.value
                        self.comm.bsend(dummy_data, dest=dest,
                                        tag=JobType.TIMEUP.value)
            while True:
                ret = self.comm.Iprobe(source=MPI.ANY_SOURCE,
                                        tag=MPI.ANY_TAG, status=status)
                if ret == False:
                    break
                else:
                    message = self.comm.recv(
                        source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    cur_status = status
                    tag = cur_status.Get_tag()
                    job = (tag, message)
                    if JobType.is_high_priority(tag):
                        # high priority messages (timeup and finish)
                        jobq.append(job)
                    else:
                        # normal messages (search and backpropagate)
                        jobq.appendleft(job)

            jobq_non_empty = bool(jobq)
            if jobq_non_empty:
                (tag, message) = jobq.pop()
                if tag == JobType.SEARCH.value:
                    # if node is not in the hash table
                    if self.hsm.search_table(message[0]) is None:
                        node = Tree_Node(state=message[0], reward_calculator=self.reward_calculator, conf=self.conf)
                        #node.state = message[0]
                        if node.state == ['&']:
                            node.expansion(self.chem_model)
                            m = random.choice(node.expanded_nodes)
                            n = node.addnode(m)
                            self.hsm.insert(Item(node.state, node))
                            _, dest = self.hsm.hashing(n.state)
                            self.send_message(n, dest, tag=JobType.SEARCH.value)
                        else:
                            # or max_len_wavelength :
                            if len(node.state) < node.max_len:
                                values_list, score, mol, filter_flag = node.simulation(
                                    self.chem_model, node.state, self.rank, gau_id)
                                gau_id += 1

                                self.record_result(smiles=mol, depth=len(node.state), reward=score,
                                                   raw_reward_list=values_list, filter_flag=filter_flag)
                                # backpropagation on local memory
                                node.update_local_node(score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)
                            else:
                                score = -1
                                # backpropagation on local memory
                                node.update_local_node(node, score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)

                    else:  # if node already in the local hashtable
                        node = self.hsm.search_table(message[0])
                        #print("debug:", node.visits,
                        #      node.num_thread_visited, node.wins)
                        if node.state == ['&']:
                            if node.expanded_nodes != []:
                                m = random.choice(node.expanded_nodes)
                                n = node.addnode(m)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(n.state)
                                self.comm.bsend(np.asarray([n.state, n.reward, n.wins, n.visits,
                                                n.num_thread_visited]), dest=dest, tag=JobType.SEARCH.value)
                            else:
                                ind, childnode = node.selection()
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(childnode.state)
                                self.send_message(childnode, dest, tag=JobType.SEARCH.value)
                        else:
                            #node.num_thread_visited = message[4]
                            if len(node.state) < node.max_len:
                                if node.state[-1] != '\n':
                                    if node.expanded_nodes != []:
                                        m = random.choice(node.expanded_nodes)
                                        n = node.addnode(m)
                                        self.hsm.insert(Item(node.state, node))
                                        _, dest = self.hsm.hashing(n.state)
                                        self.send_message(n, dest, tag=JobType.SEARCH.value)
                                    else:
                                        if node.check_childnode == []:
                                            node.expansion(self.chem_model)
                                            m = random.choice(
                                                node.expanded_nodes)
                                            n = node.addnode(m)
                                            self.hsm.insert(Item(node.state, node))
                                            _, dest = self.hsm.hashing(n.state)
                                            self.send_message(n, dest, tag=JobType.SEARCH.value)
                                        else:
                                            ind, childnode = node.selection()
                                            self.hsm.insert(Item(node.state, node))
                                            _, dest = self.hsm.hashing(
                                                childnode.state)
                                            self.send_message(childnode, dest, tag=JobType.SEARCH.value)

                                else:
                                    values_list, score, mol, filter_flag = node.simulation(
                                        self.chem_model, node.state, self.rank, gau_id)
                                    gau_id += 1
                                    score = -1
                                    self.record_result(smiles=mol, depth=len(node.state), reward=score,
                                                       raw_reward_list=[score], filter_flag=filter_flag)
                                    # backpropagation on local memory
                                    node.update_local_node(score)
                                    self.hsm.insert(Item(node.state, node))
                                    _, dest = self.hsm.hashing(node.state[0:-1])
                                    self.send_backprop(node, dest)

                            else:
                                score = -1
                                # backpropagation on local memory
                                node.update_local_node(score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)

                elif tag == JobType.BACKPROPAGATION.value:
                    node = Tree_Node(state=message[0], reward_calculator=self.reward_calculator, conf=self.conf)
                    node.reward = message[1]
                    local_node = self.hsm.search_table(message[0][0:-1])
                    if local_node.state == ['&']:
                        local_node.backpropagation(node)
                        self.hsm.insert(Item(local_node.state, local_node))
                        _, dest = self.hsm.hashing(local_node.state)
                        self.send_message(local_node, dest, tag=JobType.SEARCH.value)
                    else:
                        local_node.backpropagation(node)
                        self.hsm.insert(Item(local_node.state, local_node))
                        _, dest = self.hsm.hashing(local_node.state[0:-1])
                        self.send_backprop(local_node, dest)
                elif tag == JobType.TIMEUP.value:
                    timeup = True

        return

    def TDS_df_UCT(self):
        # self.comm.barrier()
        status = MPI.Status()
        gau_id = 0  # this is used for wavelength
        self.start_time = time.time()
        bpm = 0
        bp = []
        _, rootdest = self.hsm.hashing(['&'])
        jobq = deque()
        timeup = False
        if self.rank == rootdest:
            root_job_message = np.asarray([['&'], None, 0, 0, 0, []])
            for i in range(3 * self.nprocs):
                temp = deepcopy(root_job_message)
                root_job = (JobType.SEARCH.value, temp)
                jobq.appendleft(root_job)
        while not timeup:
            if self.rank == 0:
                if self.elapsed_time() > 600:
                    timeup = True
                    for dest in range(1, self.nprocs):
                        dummy_data = tag = JobType.TIMEUP.value
                        self.comm.bsend(dummy_data, dest=dest,
                                        tag=JobType.TIMEUP.value)
            while True:
                ret = self.comm.Iprobe(source=MPI.ANY_SOURCE,
                                        tag=MPI.ANY_TAG, status=status)
                if ret == False:
                    break
                else:
                    message = self.comm.recv(
                        source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    cur_status = status
                    tag = cur_status.Get_tag()
                    job = (tag, message)
                    if JobType.is_high_priority(tag):
                        jobq.append(job)
                    else:
                        jobq.appendleft(job)
            jobq_non_empty = bool(jobq)
            if jobq_non_empty:
                (tag, message) = jobq.pop()
                if tag == JobType.SEARCH.value:
                    if self.hsm.search_table(message[0]) == None:
                        node = Tree_Node(state=message[0], reward_calculator=self.reward_calculator, conf=self.conf)
                        info_table = message[5]
                        #print ("not in table info_table:",info_table)
                        if node.state == ['&']:
                            node.expansion(self.chem_model)
                            m = random.choice(node.expanded_nodes)
                            n = node.addnode(m)
                            self.hsm.insert(Item(node.state, node))
                            _, dest = self.hsm.hashing(n.state)
                            self.send_message(n, dest, tag=JobType.SEARCH.value)
                        else:
                            if len(node.state) < node.max_len:
                                values_list, score, mol, filter_flag = node.simulation(
                                    self.chem_model, node.state, self.rank, gau_id)
                                #print (mol)
                                gau_id += 1
                                self.record_result(smiles=mol, depth=len(node.state), reward=score,
                                                   raw_reward_list=values_list, filter_flag=filter_flag)
                                node.update_local_node(score)
                                # update infor table
                                info_table = backtrack_tdsdfuct(
                                    info_table, score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)
                            else:
                                score = -1
                                node.update_local_node(node, score)
                                info_table = backtrack_tdsdfuct(
                                    info_table, score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)

                    else:  # if node already in the local hashtable
                        node = self.hsm.search_table(message[0])
                        info_table = message[5]
                        #print ("in table info_table:",info_table)
                        if node.state == ['&']:
                            # print ("in table root:",node.state,node.path_ucb,len(node.state),len(node.path_ucb))
                            if node.expanded_nodes != []:
                                m = random.choice(node.expanded_nodes)
                                n = node.addnode(m)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(n.state)
                                self.send_message(n, dest, tag=JobType.SEARCH.value)
                            else:
                                ind, childnode = node.selection()
                                self.hsm.insert(Item(node.state, node))
                                info_table = update_selection_ucbtable_tdsdfuct(
                                    info_table, node, ind)
                                #print ("info_table after selection:",info_table)
                                _, dest = self.hsm.hashing(childnode.state)
                                self.send_message(childnode, dest, tag=JobType.SEARCH.value)
                        else:
                            #node.path_ucb = message[5]
                            # info_table=message[5]
                            #print("check ucb:", node.reward, node.visits, node.num_thread_visited,info_table)
                            if len(node.state) < node.max_len:
                                if node.state[-1] != '\n':
                                    if node.expanded_nodes != []:
                                        m = random.choice(node.expanded_nodes)
                                        n = node.addnode(m)
                                        self.hsm.insert(Item(node.state, node))
                                        _, dest = self.hsm.hashing(n.state)
                                        self.send_message(n, dest, tag=JobType.SEARCH.value)
                                    else:
                                        if node.check_childnode == []:
                                            node.expansion(self.chem_model)
                                            m = random.choice(
                                                node.expanded_nodes)
                                            n = node.addnode(m)
                                            self.hsm.insert(Item(node.state, node))
                                            _, dest = self.hsm.hashing(n.state)
                                            self.send_message(n, dest, tag=JobType.SEARCH.value)
                                        else:
                                            ind, childnode = node.selection()
                                            self.hsm.insert(Item(node.state, node))
                                            info_table = update_selection_ucbtable_tdsdfuct(
                                                info_table, node, ind)
                                            _, dest = self.hsm.hashing(
                                                childnode.state)
                                            self.send_message(childnode, dest, tag=JobType.SEARCH.value)
                                else:
                                    value_list, score, mol, filter_flag = node.simulation(
                                        self.chem_model, node.state, self.rank, gau_id)
                                    gau_id += 1
                                    score = -1
                                    self.record_result(smiles=mol, depth=len(node.state), reward=score,
                                                       raw_reward_list=value_list, filter_flag=filter_flag)
                                    node.update_local_node(score)
                                    info_table = backtrack_tdsdfuct(
                                        info_table, score)

                                    self.hsm.insert(Item(node.state, node))
                                    _, dest = self.hsm.hashing(node.state[0:-1])
                                    self.send_backprop(node, dest)
                            else:
                                score = -1
                                node.update_local_node(score)
                                info_table = backtrack_tdsdfuct(
                                    info_table, score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)

                elif tag == JobType.BACKPROPAGATION.value:
                    bpm += 1
                    node = Tree_Node(state=message[0], reward_calculator=self.reward_calculator, conf=self.conf)
                    node.reward = message[1]
                    local_node = self.hsm.search_table(message[0][0:-1])
                    #print ("report check message[5]:",message[5])
                    #print ("check:",len(message[0]), len(message[5]))
                    #print ("check:",local_node.wins, local_node.visits, local_node.num_thread_visited)
                    info_table=message[5]
                    if local_node.state == ['&']:
                        local_node.backpropagation(node)
                        self.hsm.insert(Item(local_node.state, local_node))
                        _, dest = self.hsm.hashing(local_node.state)
                        self.send_message(local_node, dest, tag=JobType.SEARCH.value)
                    else:
                        local_node.backpropagation(node)
                        #local_node,info_table = backtrack_tdsdf(info_table,local_node, node)
                        back_flag = compare_ucb_tdsdfuct(info_table,local_node)
                        self.hsm.insert(Item(local_node.state, local_node))
                        if back_flag == 1:
                            _, dest = self.hsm.hashing(local_node.state[0:-1])
                            self.send_backprop(local_node, dest)
                        if back_flag == 0:
                            _, dest = self.hsm.hashing(local_node.state)
                            self.send_message(local_node, dest, tag=JobType.SEARCH.value)
                elif tag == JobType.TIMEUP.value:
                    timeup = True
        bp.append(bpm)

        return

    def MP_MCTS(self):
        #self.comm.barrier()
        status = MPI.Status()
        gau_id = 0 ## this is used for wavelength
        self.start_time = time.time()
        _, rootdest = self.hsm.hashing(['&'])
        jobq = deque()
        timeup = False
        if self.rank == rootdest:
            root_job_message = np.asarray([['&'], None, 0, 0, 0, []])
            for i in range(3 * self.nprocs):
                temp = deepcopy(root_job_message)
                root_job = (JobType.SEARCH.value, temp)
                jobq.appendleft(root_job)
        while not timeup:
            if self.rank == 0:
                if self.elapsed_time() > 60: # default 60s
                    timeup = True
                    for dest in range(1, self.nprocs):
                        dummy_data = tag = JobType.TIMEUP.value
                        self.comm.bsend(dummy_data, dest=dest, tag=JobType.TIMEUP.value)

            while True:
                ret = self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                if ret == False:
                    break
                else:
                    message = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    cur_status = status
                    tag = cur_status.Get_tag()
                    job = (tag, message)
                    if JobType.is_high_priority(tag):
                        jobq.append(job)
                    else:
                        jobq.appendleft(job)
            jobq_non_empty = bool(jobq)
            if jobq_non_empty:
                (tag, message) = jobq.pop()
                if tag == JobType.SEARCH.value:
                    if self.hsm.search_table(message[0]) == None:
                        node = Tree_Node(state=message[0], reward_calculator=self.reward_calculator, conf=self.conf)
                        if node.state == ['&']:
                            node.expansion(self.chem_model)
                            m = random.choice(node.expanded_nodes)
                            n = node.addnode(m)
                            self.hsm.insert(Item(node.state, node))
                            _, dest = self.hsm.hashing(n.state)
                            self.send_message(n, dest, tag=JobType.SEARCH.value)
                        else:
                            if len(node.state) < node.max_len:
                                values_list, score, mol, filter_flag = node.simulation(self.chem_model, node.state, self.rank, gau_id)
                                gau_id+=1
                                self.record_result(smiles=mol, depth=len(node.state), reward=score,
                                                   raw_reward_list=values_list, filter_flag=filter_flag)
                                node.update_local_node(score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)
                            else:
                                score = -1
                                node.update_local_node(node, score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)

                    else:  # if node already in the local hashtable
                        node = self.hsm.search_table(message[0])
                        if node.state == ['&']:
                            # print ("in table root:",node.state,node.path_ucb,len(node.state),len(node.path_ucb))
                            if node.expanded_nodes != []:
                                m = random.choice(node.expanded_nodes)
                                n = node.addnode(m)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(n.state)
                                self.send_message(n, dest, tag=JobType.SEARCH.value)
                            else:
                                ind, childnode = node.selection()
                                self.hsm.insert(Item(node.state, node))
                                ucb_table = update_selection_ucbtable_mpmcts(node, ind)
                                _, dest = self.hsm.hashing(childnode.state)
                                self.send_message(childnode, dest, tag=JobType.SEARCH.value, data=ucb_table)
                        else:
                            node.path_ucb = message[5]
                            #print("check ucb:", node.wins, node.visits, node.num_thread_visited)
                            if len(node.state) < node.max_len:
                                if node.state[-1] != '\n':
                                    if node.expanded_nodes != []:
                                        m = random.choice(node.expanded_nodes)
                                        n = node.addnode(m)
                                        self.hsm.insert(Item(node.state, node))
                                        _, dest = self.hsm.hashing(n.state)
                                        self.send_message(n, dest, tag=JobType.SEARCH.value)
                                    else:
                                        if node.check_childnode == []:
                                            node.expansion(self.chem_model)
                                            m = random.choice(node.expanded_nodes)
                                            n = node.addnode(m)
                                            self.hsm.insert(Item(node.state, node))
                                            _, dest = self.hsm.hashing(n.state)
                                            self.send_message(n, dest, tag=JobType.SEARCH.value)
                                        else:
                                            ind, childnode = node.selection()
                                            self.hsm.insert(Item(node.state, node))
                                            ucb_table = update_selection_ucbtable_mpmcts(node, ind)
                                            _, dest = self.hsm.hashing(childnode.state)
                                            self.send_message(childnode, dest, tag=JobType.SEARCH.value, data=ucb_table)
                                else:
                                    values_list, score, mol, filter_flag = node.simulation(self.chem_model, node.state, self.rank, gau_id)
                                    gau_id+=1
                                    score = -1
                                    self.record_result(smiles=mol, depth=len(node.state), reward=score,
                                                       raw_reward_list=values_list, filter_flag=filter_flag)
                                    node.update_local_node(score)
                                    self.hsm.insert(Item(node.state, node))
                                    _, dest = self.hsm.hashing(node.state[0:-1])
                                    self.send_backprop(node, dest)
                            else:
                                score = -1
                                node.update_local_node(score)
                                self.hsm.insert(Item(node.state, node))
                                _, dest = self.hsm.hashing(node.state[0:-1])
                                self.send_backprop(node, dest)

                elif tag == JobType.BACKPROPAGATION.value:
                    node = Tree_Node(state=message[0], reward_calculator=self.reward_calculator, conf=self.conf)
                    node.reward = message[1]
                    local_node = self.hsm.search_table(message[0][0:-1])
                    if local_node.state == ['&']:
                        local_node.backpropagation(node)
                        self.hsm.insert(Item(local_node.state, local_node))
                        _, dest = self.hsm.hashing(local_node.state)
                        self.send_message(local_node, dest, tag=JobType.SEARCH.value)
                    else:
                        local_node.backpropagation(node)
                        local_node = backtrack_mpmcts(local_node, node)
                        back_flag = compare_ucb_mpmcts(local_node)
                        self.hsm.insert(Item(local_node.state, local_node))
                        if back_flag == 1:
                            _, dest = self.hsm.hashing(local_node.state[0:-1])
                            self.send_backprop(local_node, dest)
                        if back_flag == 0:
                            _, dest = self.hsm.hashing(local_node.state)
                            self.send_message(local_node, dest, tag=JobType.SEARCH.value)

                elif tag == JobType.TIMEUP.value:
                    timeup = True

        return
