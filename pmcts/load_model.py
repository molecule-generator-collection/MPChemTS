import sys

from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import GRU
from keras.layers.embeddings import Embedding


def get_model_structure_info(model_json):
    with open(model_json, 'r') as f:
        loaded_model_json = f.read()
    loaded_model = model_from_json(loaded_model_json)
    print(f"Loaded model_json from {model_json}")
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
        print('Confirm if the version of Tensorflow is 2.5. If so, please consult with ChemTSv2 developers on the GitHub repository. At that time, please attach the file specified as `model_json`')
        sys.exit()
            
    return input_shape, vocab_size, output_size


def loaded_model(conf):
    vocab_size=conf['rnn_vocab_size']
    embed_size=conf['rnn_output_size']

    model = Sequential()

    model.add(Embedding(input_dim=vocab_size, output_dim=vocab_size, batch_size=1, mask_zero=False))
    model.add(GRU(256, batch_input_shape=(1, None, 64), activation='tanh', return_sequences=True, stateful=True))
    model.add(GRU(256, activation='tanh', return_sequences=False, stateful=True))
    model.add(Dense(embed_size, activation='softmax'))
    model.load_weights(conf['model_weight'])
    return model
