from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers.embeddings import Embedding


def stateful_logp_model():
    vocab_size=64 #len(vocabulary)
    embed_size=64 #len(vocabulary)

    model = Sequential()

    model.add(Embedding(input_dim=vocab_size, output_dim=vocab_size, batch_size=1, mask_zero=False))
    model.add(GRU(256, batch_input_shape=(1, None, 64), activation='tanh', return_sequences=True, stateful=True))
    model.add(GRU(256, activation='tanh', return_sequences=False, stateful=True))
    model.add(Dense(embed_size, activation='softmax'))
    model.load_weights('model/model.tf25.best.ckpt.h5')
    return model
