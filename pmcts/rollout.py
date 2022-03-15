import numpy as np

"""Sampling molecules in simulation step"""
def chem_kn_simulation(model, state, val, smiles_max_len):  # MEMO: this function in ChemTSv2 deal with all added nodes (atoms)
    all_posible = []
    end = "\n"
    position = []
    position.extend(state)
    get_int = [val.index(position[j]) for j in range(len(position))]
    x = np.reshape(get_int, (1, len(get_int)))
    model.reset_states()
    while not get_int[-1] == val.index(end):
        preds = model.predict_on_batch(x)
        state_pred = np.squeeze(preds)
        next_int = np.random.choice(range(len(state_pred)), p=state_pred)
        get_int.append(next_int)
        x = np.reshape([next_int], (1, 1))
        if len(get_int) > smiles_max_len:
            break
    all_posible.append(get_int)
    return all_posible


def predict_smile(all_posible, val):
    new_compound = []
    for i in range(len(all_posible)):
        total_generated = all_posible[i]
        generate_smile = []
        for j in range(len(total_generated) - 1):
            generate_smile.append(val[total_generated[j]])
        generate_smile.remove("&")
        new_compound.append(generate_smile)
    return new_compound


def make_input_smile(generate_smile):
    new_compound = []
    for i in range(len(generate_smile)):
        middle = []
        for j in range(len(generate_smile[i])):
            middle.append(generate_smile[i][j])
        com = ''.join(middle)
        new_compound.append(com)
    return new_compound
