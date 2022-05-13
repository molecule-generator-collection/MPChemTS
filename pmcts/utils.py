import numpy as np

"""Sampling molecules in simulation step"""
def chem_kn_simulation(model, state, val, conf):  # Can be merged with ChemTSv2
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
        if len(get_int) > conf['max_len']:
            break
    return get_int


def build_smiles_from_tokens(all_posible, val):  # Can be merged with ChemTSv2
    total_generated = all_posible
    generate_tokens = [val[total_generated[j]] for j in range(len(total_generated) - 1)]
    generate_tokens.remove("&")
    return ''.join(generate_tokens)
