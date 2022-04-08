import argparse
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import pickle
import random
import yaml

import numpy as np
from rdkit import RDLogger

from mpi4py import MPI
from pmcts.load_model import stateful_logp_model
from pmcts.zobrist_hash import HashTable
from pmcts.search_tree import Tree_Node
from pmcts.parallel_mcts import p_mcts


def get_parser():
    parser = argparse.ArgumentParser(
        description="",
        usage=f"python {os.path.basename(__file__)} -c CONFIG_FILE"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True,
        help="path to a config file"
    )
    parser.add_argument(
        "-d", "--debug", action='store_true',
        help="debug mode"
    )
    parser.add_argument(
        "-g", "--gpu", type=str,
        help="constrain gpu. (e.g. 0,1)"
    )
    return parser.parse_args()
    

def set_default_config(conf):
    conf.setdefault('output_dir', 'result')
    conf.setdefault('property', 'logP')
    conf.setdefault('random_seed', 3)
    conf.setdefault('token', 'model/tokens.pkl')


if __name__ == "__main__":
    args = get_parser()
    with open(args.config, 'r') as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)
    
    conf['debug'] = args.debug
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if args.gpu is None else args.gpu
    if not conf['debug']:
        RDLogger.DisableLog("rdApp.*")

    with open(conf['token'], 'rb') as f:
        tokens = pickle.load(f)
    conf['token'] = tokens

    print(f"========== Configuration ==========")
    for k, v in conf.items():
        print(f"{k}: {v}")
    print(f"GPU devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"===================================")

    """
    Initialize MPI environment
    """
    print("Initialize MPI environment")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    status = MPI.Status()
    mem = np.zeros(1024 * 10 * 1024)
    MPI.Attach_buffer(mem)

    """
    load the pre-trained rnn model and define the property optimized:
    currently available properties: logP (rdkit) and wavelength (DFT)
    """
    print('load the pre-trained rnn model and define the property optimized')
    chem_model = stateful_logp_model()
    property = conf['property']
    node = Tree_Node(state=['&'], property=property, conf=conf)

    """
    Initialize HashTable
    """
    print('Initialize HashTable')
    random.seed(conf['random_seed'])
    hsm = HashTable(nprocs, node.val, node.max_len, len(node.val))

    """
    Design molecules using parallel MCTS: TDS-UCT,TDS-df-UCT and MP-MCTS
    """
    print('Run MPChemTS')
    comm.barrier()
    #score,mol=p_mcts.TDS_UCT(chem_model, hsm, property, comm, conf)
    #score,mol=p_mcts.TDS_df_UCT(chem_model, hsm, property, comm, conf)
    score, mol = p_mcts.MP_MCTS(chem_model, hsm, property, comm, conf)

    print("Done MCTS execution")

    result = list(map(lambda x, y:(x,y), score, mol))
    result = sorted(result, key = lambda x: x[0], reverse=True)
    result = result[:3]
    comm.barrier()

    if rank==0:
      for src in range(1, nprocs):
        data = comm.recv(source=src, tag=999, status=status)
        result.extend(data)
      result = sorted(result, key = lambda x: x[0], reverse=True)
      result = result[:3]
      max_reward = result[0][0]
      plogp_score = max_reward/(1.0 - max_reward)
      print("\nTop 3 molecules\nreward\tplogp score\tmolecule SMILES")
      for itr in result:
        reward = itr[0]
        plogp_score = reward/(1.0 - reward) 
        print(str(round(reward, 4))+'\t'+str(round(plogp_score, 4))+'\t\t'+str(itr[1]))
#      print("max reward: %.4f, max plogp score: %.4f\n" %(max_reward, plogp_score))
    else:
      comm.send(result, dest = 0, tag=999)


    output_score_path = os.path.join(conf['output_dir'], f"logp_score{rank}.csv")
    with open(output_score_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\n')
        writer.writerow(score)
    output_mol_path = os.path.join(conf['output_dir'], f"logp_mol{rank}.csv")
    with open(output_mol_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\n')
        writer.writerow(mol)
