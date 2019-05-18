import tensorflow as tf
from random_words import RandomWords
import numpy as np
import os
import pyphen
import matplotlib.pyplot as plt
from utils import preprocess

tf.enable_eager_execution()
path_to_file = './data/eng-haiku.txt'

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
text = preprocess(text)

# length of text is the number of characters in it
print('Length of text: {} characters'.format(len(text)))
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# The maximum length sentence we want for a single input in characters
seq_length = 20
examples_per_epoch = len(text)//seq_length
# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 100000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1048

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

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)


for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)


def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    period=8500)

EPOCHS=1000000

#########UNCOMMENT TO TRAIN############
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback], steps_per_epoch=1)

fig, ax1 = plt.subplots()
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Loss")
ax1.plot(list(range(1, len(history.history['loss']) + 1)), history.history['loss'], 'b', label='Loss')

# ax2 = ax1.twinx()
# ax2.set_ylabel("Acurracy")
# ax2.plot(list(range(1, len(history.history['acc']) + 1)), history.history['acc'], 'r', label='Accuracy')

fig.legend(loc='best')
plt.title("Training")

plt.show()
plt.savefig("wykres.png")
plt.close()

#########UNCOMMENT TO LOAD############
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string, num_generate=100):
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
  temperature = 0.6

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

  return (start_string + ''.join(text_generated))


# def generate_haiku(model):
#     haiku = 0
#     five_syllables = []
#     seven_syllables = []
#
#     while haiku == 0:
#         text = generate_text(model,RandomWords().random_word())
#         text = text.replace('<br>', '\n').replace('<br','\n').replace('br>','\n').split('\n')
#         dic = pyphen.Pyphen(lang='en_EN')
#
#         for buf in text:
#             syllables = 0
#             for word in buf.split(' '):
#                 if len(word) > 0:
#                     syllables += dic.inserted(word).count('-')+1
#             if syllables == 5:
#                 five_syllables.append(buf+'\n')
#             elif syllables == 7:
#                 seven_syllables.append(buf+'\n')
#
#         if len(seven_syllables) >= 2 and len(five_syllables) >= 1:
#             haiku = seven_syllables[0]+five_syllables[0]+seven_syllables[1]
#
#     return haiku


def generate_haiku(model):
    result = generate_text(model, start_string='\n', num_generate=100)
    endlines = 0
    for i, char in enumerate(result):
        if char == '\n':
            endlines += 1
        if endlines == 4:
            return result[1:i]

#print(generate_text(model, start_string=u"Flowers"))
print("Generated haiku 1. :")
print(generate_haiku(model))

print("Generated haiku 2. :")
print(generate_haiku(model))

print("Generated haiku 3. :")
print(generate_haiku(model))
