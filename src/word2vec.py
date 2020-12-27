import tensorflow as tf
import numpy as np
import collections
import random
from matplotlib import pyplot as plt
from scipy.spatial import distance
from src.midi import MIDIModule
import math
from music21 import chord
'''
Word2Vec model for creating a music vocabulary.
Inspiration: Modeling Musical Context with Word2vec by Dorien Herremans, Ching-Hua Chuan
'''


def build_vocabulary(directory, make_chords_from_notes = False):
    '''
    Create a vocabulary (dict and reverse dictionary) based on music in a directory.
    Input:
    directory: where to get music from
    Output:
    word_dict: note to integer mapping
    reverse_dictionary: integer to note mapping
    mapped_data: mapping to int of every note
    '''
    data = MIDIModule.get_notes(directory, make_chords_from_notes)
    word_dict = MIDIModule.to_int(data)
    reverse_dictionary = dict(zip(word_dict.values(),word_dict.keys()))
    mapped_data = []
    for element in data:
        mapped_data.append(word_dict[element])
    return word_dict, reverse_dictionary, mapped_data



def generate_batch(batch_size, num_skips, skip_window, data):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window

  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

  span = 2 * skip_window + 1  # [ skip_window target skip_window ]  Context that you predict

  buffer = collections.deque(maxlen=span)

  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)

  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels


# Step 4: Build and train a skip-gram model.
directory ="" #TODO: directory here
dictionary, reverse_dictionary, data = build_vocabulary(directory)
vocabulary_size = len(directory)
batch_size = 128
embedding_size = 256  # Dimension of the embedding vector.
skip_window = 4  # How many words to consider left and right.
num_skips = skip_window * 2  # How many times to reuse an input to generate a label.
alpha = 0.1  # learning rate
num_steps = 1000001  # how long to train for

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 5  # Random set of words to evaluate similarity on.
valid_window = 20  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64  # Number of negative examples to sample.

# Hand-picked validation set for music
fMajor = chord.Chord("F A C")
cMajor = chord.Chord("C E G")
gMajor = chord.Chord("G B D")

valid_centers = [cMajor, fMajor, gMajor]  # 145: C-E-G, 545: F-A-C, 2180: G-B-D
valid_neighbors = [[fMajor, gMajor, chord.Chord('A C E'), chord.Chord("G Bb Eb"), chord.Chord("F A D"), chord.Chord('G Bb D')],
                   [cMajor, chord.Chord("Bb D F"), chord.Chord("D F A"),chord.Chord("C Eb Ab"), chord.Chord("Bb D G"),chord.Chord("C Eb G")],
                   [chord.Chord("D F# A"),cMajor,chord.Chord("E G B"), chord.Chord("D F Bb"), chord.Chord("C E A"), chord.Chord("D F A")]] # V, IV, vi, IIIb, IIb, v

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    # with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random.uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embeddings')
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.random.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # Construct a ADAM optimizer.
    optimizer = tf.optimizers.Adam(alpha)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True), name='norm')
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True, name='similarity')

    # Add variable initializer.
    init = tf.global_variables_initializer()

# Step 5: Begin training.
with tf.compat.v1.Session(graph=graph) as session:
    saver = tf.train.compat.v1.Saver()
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")
    average_loss = 0
    results_C = []
    results_F = []
    results_G = []
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            # print("Average loss at step ", step, ": ", average_loss)
            print(average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        # get the 8  most closest to the validation set
        if step % 10000 == 0:
            sim = similarity.eval()
            norm_embeddings = normalized_embeddings.eval()

            # hand-picked validation examples
            for i, center in enumerate(valid_centers):
                center_i = dictionary[center]
                sim_values = []
                log_str = "Similarity to %s:" % dictionary[center]
                for neighbor in valid_neighbors[i]:
                    neighbor_i = dictionary[neighbor]
                    center_embedding = norm_embeddings[center_i]
                    neighbor_embedding = norm_embeddings[neighbor_i]
                    cos_dist = distance.cosine(center_embedding, neighbor_embedding)
                    sim_values.append(cos_dist)
                    log_str = "%s %s:%s," % (log_str, dictionary[neighbor], cos_dist)
                if i == 0:
                    results_C.append(sim_values)
                elif i == 1:
                    results_F.append(sim_values)
                else:
                    results_G.append(sim_values)
                print(log_str)

    final_embeddings = normalized_embeddings.eval()
    saver.save(session, "saves/word2vec_music_pc_train")
    np.savetxt('results_C.txt', results_C, delimiter=',', newline='\n')
    np.savetxt('results_F.txt', results_F, delimiter=',', newline='\n')
    np.savetxt('results_G.txt', results_G, delimiter=',', newline='\n')



def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)