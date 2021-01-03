import tensorflow as tf
import numpy as np
import collections
import random
from matplotlib import pyplot as plt
from scipy.spatial import distance
from src.midi import MIDIModule
from src.positional_encoding import positional_encoding
import math
from music21 import chord
from music21.alpha.analysis.hasher import Hasher
'''
Word2Vec model for creating a music vocabulary.
Inspiration: Modeling Musical Context with Word2vec by Dorien Herremans, Ching-Hua Chuan
'''



class word2vec_music(tf.keras.layers.Layer):
    def __init__(self, graph = None):
        super(word2vec_music, self).__init__()
        self.embeddings = None
        if graph == None:
            self.graph = tf.Graph()
        else:
            self.graph = graph

    def build_vocabulary(self, directory, make_chords_from_notes=False):
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
        print(data)
        word_dict = MIDIModule.to_int(data)
        print(word_dict)
        print(len(word_dict))

        reverse_dictionary = dict(zip(word_dict.values(), word_dict.keys()))
        mapped_data = []
        for element in data:
            mapped_data.append(word_dict[element])
        return word_dict, reverse_dictionary, mapped_data



    def generate_batch(self, batch_size, num_skips, skip_window, data):
        data_index = 0
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
    def train_word2vec(self, directory, embedding_size=256, num_steps=1000001):
        dictionary, reverse_dictionary, data = self.build_vocabulary(directory)
        vocabulary_size = len(dictionary)
        batch_size = 8  # TODO:change this to 128 later
        embedding_size = embedding_size  # Dimension of the embedding vector.
        skip_window = 1  # How many words to consider left and right.
        num_skips = skip_window * 2  # How many times to reuse an input to generate a label.
        alpha = 0.01  # learning rate
        num_steps = num_steps  # how long to train for

        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        # TODO: scale these back up
        valid_size = 2  # Random set of words to evaluate similarity on.
        valid_window = 8  # Only pick dev samples in the head of the distribution.
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        num_sampled = 16  # Number of negative examples to sample.

        # Hand-picked validation set for music

        # TODO: if we choose to use chords as validation, have to change them to string format to the mapping is preserved. The chords HAVE to be present in the data.

        # fMajor = chord.Chord("F A C")
        # cMajor = chord.Chord("C E G")
        # gMajor = chord.Chord("G B D")
        #
        # valid_centers = [cMajor, fMajor, gMajor]  # 145: C-E-G, 545: F-A-C, 2180: G-B-D
        # valid_neighbors = [[fMajor, gMajor, chord.Chord('A C E'), chord.Chord("G Bb Eb"), chord.Chord("F A D"), chord.Chord('G Bb D')],
        #                    [cMajor, chord.Chord("Bb D F"), chord.Chord("D F A"),chord.Chord("C Eb Ab"), chord.Chord("Bb D G"),chord.Chord("C Eb G")],
        #                    [chord.Chord("D F# A"),cMajor,chord.Chord("E G B"), chord.Chord("D F Bb"), chord.Chord("C E A"), chord.Chord("D F A")]] # V, IV, vi, IIIb, IIb, v


        with self.graph.as_default():
            # Input data.
            train_inputs = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            # with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random.uniform([vocabulary_size + 1, embedding_size], -1.0, 1.0), name='embeddings')
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

            # Construct an ADAm optimizer.
            optimizer = tf.compat.v1.train.AdamOptimizer(alpha).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True), name='norm')
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True, name='similarity')

            # Add variable initializer.
            init = tf.compat.v1.global_variables_initializer()

        # Step 5: Begin training.
        with tf.compat.v1.Session(graph=self.graph) as session:
            saver = tf.compat.v1.train.Saver()
            # We must initialize all variables before we use them.
            init.run()
            print("Initialized")
            average_loss = 0
            results_C = []
            results_F = []
            results_G = []
            for step in range(num_steps):
                batch_inputs, batch_labels = self.generate_batch(
                    batch_size, num_skips, skip_window, data)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step ", step, ": ", average_loss)
                    # print(average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                # get the 8  most closest to the validation set
            #     if step % 10000 == 0:
            #         sim = similarity.eval()
            #         norm_embeddings = normalized_embeddings.eval()
            #
            #         # hand-picked validation examples
            #         for i, center in enumerate(valid_centers):
            #             center_i = dictionary[Hasher(center)]
            #             sim_values = []
            #             log_str = "Similarity to %s:" % dictionary[center]
            #             for neighbor in valid_neighbors[i]:
            #                 neighbor_i = dictionary[neighbor]
            #                 center_embedding = norm_embeddings[center_i]
            #                 neighbor_embedding = norm_embeddings[neighbor_i]
            #                 cos_dist = distance.cosine(center_embedding, neighbor_embedding)
            #                 sim_values.append(cos_dist)
            #                 log_str = "%s %s:%s," % (log_str, dictionary[neighbor], cos_dist)
            #             if i == 0:
            #                 results_C.append(sim_values)
            #             elif i == 1:
            #                 results_F.append(sim_values)
            #             else:
            #                 results_G.append(sim_values)
            #             print(log_str)
            #
            final_embeddings = normalized_embeddings.eval()
            # saver.save(session, "saves/word2vec_music_pc_train")
            # np.savetxt('results_C.txt', results_C, delimiter=',', newline='\n')
            # np.savetxt('results_F.txt', results_F, delimiter=',', newline='\n')
            # np.savetxt('results_G.txt', results_G, delimiter=',', newline='\n')
            print(normalized_embeddings)
            self.embeddings = final_embeddings


    def build(self, directory, embedding_size=256, num_steps=1000001):
       self.train_word2vec(directory, embedding_size, num_steps)


    def __call__(self, input):
        with tf.compat.v1.Session(graph=self.graph) as session:
            return tf.nn.embedding_lookup(self.embeddings, input)







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


def get_embedding(elements,graph, trained_embeddings,embedding_size=256):
    """Returns Embeddings of elements.

    One-hot for notes. If the get_notes methods are used with
    make_notes_from_chords = True then it is is multi-hot for notes,
    else it is one hot

    Args:
        elements    : List of encoded elements (notes, chords, durations).
        graph : a trained embedding layer
        embedding_size: dimesion of embedding.
    Returns:
        embeddings  : positional encoding + trained embedding.
    """

    with tf.compat.v1.Session(graph=graph) as session:
        embeddings = tf.nn.embedding_lookup(params = trained_embeddings, ids = elements)
        output_list = []

        for i in range(len(elements)):
            output_list.append(tf.add(embeddings[i], positional_encoding(i, embedding_size)[:, :embedding_size, :]))
        return output_list

def get_sequences(directory, sequence_length, graph = tf.Graph(),embedding_size=256,num_steps = 1000001 ):
    """Retrieve the input and output sequences.

    Args:
        directory          : Where to get notes.
        sequence_length: dimension of embedding.
        sequence_length : Length of the sequence ( ex. len(note_list) ).

    Returns:
        network_input   : Input sequence.
        network_output  : Output sequence.
    """
    layer = word2vec_music(graph)
    layer.build(directory, embedding_size, num_steps)
    notes = MIDIModule.get_encododed_notes(directory)
    length = len(notes)
    notes = layer(notes)

    with tf.compat.v1.Session(graph=graph) as session:
        network_input = []
        network_output = []
        for i in range(0, length - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append(sequence_in)
            network_output.append(sequence_out)

    return network_input, network_output