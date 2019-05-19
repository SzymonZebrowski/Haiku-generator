import tensorflow as tf
#from random_words import RandomWords
import numpy as np
import os
import pickle
import pyphen
import matplotlib.pyplot as plt
from utils import preprocess

tf.enable_eager_execution()
path_to_file = './data/eng-haiku.txt'
syllables_data = './data/eng_haiku'
seq_length = 15
BATCH_SIZE = 64
BUFFER_SIZE = 10000
embedding_dim = 256
rnn_units = 1048
EPOCHS = 100000


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None]),
      tf.keras.layers.CuDNNLSTM(rnn_units,
                          return_sequences=True,
                          stateful=True,
                          recurrent_initializer='glorot_uniform'),
      tf.keras.layers.Dense(vocab_size)
    ])
    return model


def create_helpers(path_to_file, syllables_data):
    try:
        file = open(syllables_data, 'rb')
        text = pickle.load(file)
    except FileNotFoundError:
        text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
        text = preprocess(text)
        with open(syllables_data, 'wb') as file:
            pickle.dump(text, file)

    #print(text)
    vocab = sorted(set(text))

    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])
    return text, vocab, char2idx, idx2char, text_as_int


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def create_dataset(seq_length):
    examples_per_epoch = len(text) // seq_length
    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    return dataset


def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def train(model):
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback], steps_per_epoch=1)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.plot(list(range(1, len(history.history['loss']) + 1)), history.history['loss'], 'b', label='Loss')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Acurracy")
    ax2.plot(list(range(1, len(history.history['acc']) + 1)), history.history['acc'], 'r', label='Accuracy')

    fig.legend(loc='best')
    plt.title("Training")

    plt.show()
    plt.savefig("wykres.png")
    plt.close()
    return model


def load_latest_model():
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
    return model


def generate_text(model, start_string, temperature, num_generate=100):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    # temperature = 0.3

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)


def generate_haiku(model, temperature):
    result = generate_text(model, start_string='\n', num_generate=60, temperature=temperature)
    endlines = 0
    for i, char in enumerate(result):
        if char == '\n':
            endlines+=1
        if endlines == 4:
            return result[1:i]


text, vocab, char2idx, idx2char, text_as_int = create_helpers(path_to_file, syllables_data)
dataset = create_dataset(seq_length)

vocab_size = len(vocab)

model = build_model(
  vocab_size=len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)


for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)


example_batch_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    period=1000)


#########UNCOMMENT TO TRAIN############
#model = train(model)

#########UNCOMMENT TO LOAD############
model = load_latest_model()


#########################NOT-SO-PERFECT##############
# def generate_haiku(model):
#     haiku = 0
#     five_syllables = []
#     seven_syllables = []
#
#     while haiku == 0:
#         text = generate_text(model,RandomWords().random_word())
#         text = text.replace('<br>', '\n').replace('<br','\n').replace('br>','\n').split('\n')
#         dic = pyphen.Pyphen(lang='en')
#         for buf in text:
#             syllables = 0
#             for word in buf.split(' '):
#                 if len(word) > 0:
#                     print(word)
#                     syllables += dic.inserted(word).count('-')+1
#             if buf[0] == ' ' and len(buf) > 0:
#                 buf = buf[1:]
#             if syllables == 5:
#                 five_syllables.append(buf+'\n')
#             elif syllables == 7:
#                 seven_syllables.append(buf+'\n')
#
#         if len(seven_syllables) >= 2 and len(five_syllables) >= 1:
#             haiku = seven_syllables[0]+five_syllables[0]+seven_syllables[1]
#     return haiku
#################





#print(generate_text(model, start_string='\n'))
temperature = 0.99
print("Generated haiku 1. :")
print(generate_haiku(model, temperature))

print("Generated haiku 2. :")
print(generate_haiku(model, temperature))

print("Generated haiku 3. :")
print(generate_haiku(model, temperature))

print("Generated haiku 4. :")
print(generate_haiku(model, temperature))

print("Generated haiku 4. :")
print(generate_haiku(model, temperature))
