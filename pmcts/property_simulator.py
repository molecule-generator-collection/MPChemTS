import pickle

from pmcts import sascorer
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdmolops
import networkx as nx

class simulator:
    """
    logp property
    """
    def __init__(self, property):
        self.property=property
        if self.property=="logP":
            with open('model/tokens.pkl', 'rb') as f:
                self.val = pickle.load(f)
            self.max_len=82

    def run_simulator(self, new_compound, rank):
        if self.property=="logP":
            score,mol=self.logp_evaluator(new_compound, rank)
        return score, mol

    def logp_evaluator(self, new_compound, rank):
        ind=rank
        try:
            m = Chem.MolFromSmiles(str(new_compound[0]))
        except BaseException:
            m = None
        if m is not None:
            try:
                logp = Descriptors.MolLogP(m)
            except BaseException:
                logp = -1000
            SA_score = -sascorer.calculateScore(MolFromSmiles(new_compound[0]))
            cycle_list = nx.cycle_basis(
                nx.Graph(
                    rdmolops.GetAdjacencyMatrix(
                        MolFromSmiles(
                            new_compound[0]))))
            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([len(j) for j in cycle_list])
            if cycle_length <= 6:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 6
            cycle_score = -cycle_length
            SA_score_norm = SA_score  # (SA_score-SA_mean)/SA_std
            logp_norm = logp  # (logp-logP_mean)/logP_std
            cycle_score_norm = cycle_score  # (cycle_score-cycle_mean)/cycle_std
            score_one = SA_score_norm + logp_norm + cycle_score_norm
            score = score_one / (1 + abs(score_one))
        else:
            score = -1000 / (1 + 1000)
        return score, new_compound[0]
