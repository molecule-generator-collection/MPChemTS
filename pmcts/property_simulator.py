from rdkit import Chem


class simulator:
    def __init__(self, reward_calculator, conf):
        self.reward_calculator = reward_calculator
        self.val = conf['token']
        self.max_len=conf['max_len']

    def run_simulator(self, new_compound, conf):
        mol = Chem.MolFromSmiles(new_compound)
        if mol is None:
            score = -1000 / (1 + 1000)
        else:
            values_list = [f(mol) for f in self.reward_calculator.get_objective_functions(conf)]
            score = self.reward_calculator.calc_reward_from_objective_values(values=values_list, conf=conf)
        return score, new_compound
