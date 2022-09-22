import itertools
import sys

from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import GRU
from keras.layers.embeddings import Embedding
import numpy as np
from rdkit import Chem


def expanded_node(model, state, val, threshold=0.95):  # Can be merged with ChemTSv2
    get_int = [val.index(state[j]) for j in range(len(state))]
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
    return sorted_idxs[:i]


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


def get_model_structure_info(model_json, logger):  # Can be merged with ChemTSv2
    with open(model_json, 'r') as f:
        loaded_model_json = f.read()
    loaded_model = model_from_json(loaded_model_json)
    logger.debug(f"Loaded model_json from {model_json}")
    input_shape = None
    vocab_size = None
    output_size = None
    for layer in loaded_model.get_config()['layers']:
        config = layer.get('config')
        if layer.get('class_name') == 'InputLayer':
            input_shape = config['batch_input_shape'][1]
        if layer.get('class_name') == 'Embedding':
            vocab_size = config['input_dim']
        if layer.get('class_name') == 'TimeDistributed':
            output_size = config['layer']['config']['units']
    if input_shape is None or vocab_size is None or output_size is None:
        logger.error('Confirm if the version of Tensorflow is 2.5. If so, please consult with ChemTSv2 developers on the GitHub repository. At that time, please attach the file specified as `model_json`')
        sys.exit()
            
    return input_shape, vocab_size, output_size


def loaded_model(model_weight, logger, conf):  # Can be merged with ChemTSv2
    model = Sequential()
    model.add(Embedding(input_dim=conf['rnn_vocab_size'], output_dim=conf['rnn_vocab_size'],
                        batch_size=1, mask_zero=False))
    model.add(GRU(256, batch_input_shape=(1, None, conf['rnn_vocab_size']), activation='tanh',
                  return_sequences=True, stateful=True))
    model.add(GRU(256, activation='tanh', return_sequences=False, stateful=True))
    model.add(Dense(conf['rnn_output_size'], activation='softmax'))
    model.load_weights(model_weight)
    logger.debug(f"Loaded model_weight from {model_weight}")

    return model


def has_passed_through_filters(smiles, conf):  # Can be merged with ChemTSv2
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # default check
        return False
    checks = [f.check(mol, conf) for f in conf['filter_list']]
    return all(checks)
