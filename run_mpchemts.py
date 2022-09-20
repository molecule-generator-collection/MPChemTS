import argparse
import csv
from importlib import import_module
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import re
import sys
import pickle
import yaml

import numpy as np
from rdkit import RDLogger

from mpi4py import MPI
from pmcts.utils import loaded_model, get_model_structure_info
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
    conf.setdefault('random_seed', 3)
    conf.setdefault('token', 'model/tokens.pkl')

    conf.setdefault('model_json', 'model.tf25.json')
    conf.setdefault('model_weight', 'model/model.tf25.best.ckpt.h5')
    conf.setdefault('reward_setting', {
        'reward_module': 'reward.logP_reward',
        'reward_class': 'LogP_reward'})

    conf.setdefault('search_type', 'MP_MCTS')

    conf.setdefault('use_lipinski_filter', True)
    conf.setdefault('lipinski_filter', {
        'module': 'filter.lipinski_filter',
        'class': 'LipinskiFilter',
        'type': 'rule_of_5'})
    conf.setdefault('use_radical_filter', True)
    conf.setdefault('radical_filter', {
        'module': 'filter.radical_filter',
        'class': 'RadicalFilter'})
    conf.setdefault('use_hashimoto_filter', True) 
    conf.setdefault('hashimoto_filter', {
        'module': 'filter.hashimoto_filter',
        'class': 'HashimotoFilter'}) 
    conf.setdefault('use_sascore_filter', True)
    conf.setdefault('sascore_filter', {
        'module': 'filter.sascore_filter',
        'class': 'SascoreFilter',
        'threshold': 3.5})
    conf.setdefault('use_ring_size_filter', True)
    conf.setdefault('ring_size_filter', {
        'module': 'filter.ring_size_filter',
        'class': 'RingSizeFilter',
        'threshold': 6})

    
def get_filter_modules(conf):
    pat = re.compile(r'^use.*filter$')
    module_list = []
    for k, frag in conf.items():
        if not pat.search(k) or frag != True:
            continue
        _k = k.replace('use_', '')
        module_list.append(getattr(import_module(conf[_k]['module']), conf[_k]['class']))
    return module_list


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
    conf['max_len'], conf['rnn_vocab_size'], conf['rnn_output_size'] = get_model_structure_info(conf['model_json'])

    rs = conf['reward_setting']
    reward_calculator = getattr(import_module(rs['reward_module']), rs['reward_class'])

    print(f"========== Configuration ==========")
    for k, v in conf.items():
        print(f"{k}: {v}")
    print(f"GPU devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"===================================")

    conf['filter_list'] = get_filter_modules(conf)

    print("Initialize MPI environment")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    status = MPI.Status()
    mem = np.zeros(1024 * 10 * 1024)
    MPI.Attach_buffer(mem)

    print('load the psre-trained rnn model and define the property optimized')
    chem_model = loaded_model(conf)

    print('Run MPChemTS')
    comm.barrier()
    search = p_mcts(comm, chem_model, reward_calculator, conf)
    if conf['search_type'] == 'TDS_UCT':
        search.TDS_UCT()
    elif conf['search_type'] == 'TDS_df_UCT':
        search.TDS_df_UCT()
    elif conf['search_type'] == 'MP_MCTS':
        search.MP_MCTS()
    else:
        print('[ERROR] Select a search type from [TDS_UCT, TDS_df_UCT, MP_MCTS]')
        sys.exit(1)

    print("Done MCTS execution")

    comm.barrier()

    search.gather_results()
    if rank==0:
        search.flush()

#    output_score_path = os.path.join(conf['output_dir'], f"logp_score{rank}.csv")
#    with open(output_score_path, 'w', newline='') as f:
#        writer = csv.writer(f, delimiter='\n')
#        writer.writerow(score)
#    output_mol_path = os.path.join(conf['output_dir'], f"logp_mol{rank}.csv")
#    with open(output_mol_path, 'w', newline='') as f:
#        writer = csv.writer(f, delimiter='\n')
#        writer.writerow(mol)
