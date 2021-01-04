# Local Imports
# ...

# Third-party imports
from music21 import *
import numpy as np

class word2vec():
    """...

    ...
    """

    def __init__(self, epochs=25, n=10, alpha=1e-4, window_size=2):
        """...

        ...

        Args:
            epochs      : ...
            n           : ...
            alpha       : ...
            window_size : ...

        Returns:
            None
        """

        self.epochs = epochs
        self.n = n
        self.alpha = alpha
        self.window_size = window_size
        
    def process_data(self, data):
        """... """

        def helper(x, d):
            """If the current item is not in the dictionary, it begins with the value 
            one. Else, it increments the value. """

            if x not in d:
                d[x] = 1
            else:
                d[x] += 1

        # Generates a key-count dictionary.
        note_count = dict()
        for generalNote in data:
            generalNote = generalNote.split('.')
            
            if (len(generalNote) > 1):
                for note in generalNote:
                    helper(note, note_count)
            else:
                helper(generalNote[0], note_count)

        # Note 'vocabulary' and 'vocabulary size' are assigned to the instance.
        self.notes = sorted(sorted(note_count, key=lambda x: x[0]), key=lambda x: x[-1])
        self.n_notes = len(self.notes)
        
        # Forward and back mapping are assigned to the instance.
        self.forward_map = {note: i for (i, note) in enumerate(self.notes)}
        self.back_map = {i: note for (i, note) in enumerate(self.notes)}
                
        # Processes the data from notes and chords into n-hot vectors with context.
        processed_data = list()
        for i in range(len(data)):
            # n-hot encodes current data.
            target = self.to_nhot(data[i])

            # Determines and n-hot encodes context data.
            context = list()
            for j in range(i - self.window_size, i + self.window_size + 1):
                if ((j != i) and (j <= (len(data) - 1)) and (j >= 0)):
                    context.append(self.to_nhot(data[j]))
                    
            processed_data.append([target, context])
            
        # Returns the processed data in the form: [current_vector, [context_vectors, ...]].
        return np.array(processed_data, dtype=list)

    def to_nhot(self, generalNote):
        """...
        
        Args:
            generalNote :
            
        Returns:
            ls  :
        """

        # Parses the data; used for chords.
        generalNote = generalNote.split('.')

        # Generate a list filled with zeroes.
        ls = [0 for _ in range(self.n_notes)]
        # Hot encodes the index(es) of generalNote.
        for i in range(self.n_notes):
            for note in generalNote:
                if (i == self.forward_map[note]):
                    ls[i] = 1
        
        return ls

    def forward_prop(self, x):
        """...

        Args:
            x   :

        Returns:
            y_c :
            h   :
            u   :
        """

        # SOFTMAX ACTIVATION FUNCTION
        def softmax(x):
            e_x = np.exp(x - np.max(x))

            return e_x / e_x.sum(axis=0)

        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = softmax(u)
        return y_c, h, u

    def  back_prop(self, e, h, x):
        """...

        Args:
            e   : 
            h   :
            x   :

        Returns:
            ...
        """

        dl_dw2 = np.outer(h, e)  
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        # UPDATE WEIGHTS
        self.w1 = self.w1 - (self.alpha * dl_dw1)
        self.w2 = self.w2 - (self.alpha * dl_dw2)
        pass

    def train(self, training_data):
        """...

        ...

        Args:
            training_data   : ...

        Returns:
            ...
        """

        self.w1 = np.random.uniform(-0.8, 0.8, (self.n_notes, self.n))     # embedding matrix
        self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.n_notes))     # context matrix
        
        for i in range(0, self.epochs):

            self.loss = 0

            for n_t, n_c in training_data:

                y_pred, h, u = self.forward_prop(n_t)

                EI = np.sum([np.subtract(y_pred, note) for note in n_c], axis=0)

                self. back_prop(EI, h, n_t)

                self.loss += -np.sum([u[note.index(1)] for note in n_c]) + len(n_c) * np.log(np.sum(np.exp(u)))
                #self.loss += -2*np.log(len(n_c)) -np.sum([u[word.index(1)] for word in n_c]) + (len(n_c) * np.log(np.sum(np.exp(u))))
                
            print('EPOCH:',i, 'LOSS:', self.loss)
        pass

    def note_vec(self, note):
        """Input a word, returns a vector (if available).

        Args:
            note    :
        
        Returns:
            v_n :
        """

        n_index = self.forward_map[note]
        v_n = self.w1[n_index]
        return v_n

    def vec_sim(self, vec, top_n):
        """Input a vector, returns nearest word(s).

        Args:
            vec     :
            top_n   :
        
        Returns:
            None
        """

        note_sim = {}
        for i in range(self.n_notes):
            v_n2 = self.w1[i]
            theta_num = np.dot(vec, v_n2)
            theta_den = np.linalg.norm(vec) * np.linalg.norm(v_n2)
            theta = theta_num / theta_den

            note = self.back_map[i]
            note_sim[note] = theta

        notes_sorted = sorted(note_sim.items(), key=lambda note, sim:sim, reverse=True)

        for note, sim in notes_sorted[:top_n]:
            print(note, sim)
            
        pass

    def note_sim(self, note, top_n):
        """Input word, returns top [n] most similar words.

        Args:
            note    :
            top_n   :
        
        Returns:
            None
        """
        
        n1_index = self.forward_map[note]
        v_n1 = self.w1[n1_index]

        # CYCLE THROUGH VOCAB
        note_sim = {}
        for i in range(self.n_notes):
            v_n2 = self.w1[i]
            theta_num = np.dot(v_n1, v_n2)
            theta_den = np.linalg.norm(v_n1) * np.linalg.norm(v_n2)
            theta = theta_num / theta_den

            note = self.back_map[i]
            note_sim[note] = theta

        notes_sorted = sorted(note_sim.items(), key=lambda note, sim:sim, reverse=True)

        for note, sim in notes_sorted[:top_n]:
            print(note, sim)
            
        pass


























# import tensorflow as tf
# import numpy as np
# import collections
# import random
# from matplotlib import pyplot as plt
# from scipy.spatial import distance
# from src.midi import MIDIModule
# from src.positional_encoding import positional_encoding
# import math
# from music21 import chord
# from music21.alpha.analysis.hasher import Hasher

# '''
# Word2Vec model for creating a music vocabulary.
# Inspiration: Modeling Musical Context with Word2vec by Dorien Herremans, Ching-Hua Chuan
# '''

# def build_vocabulary(directory, make_chords_from_notes = False):
#     '''
#     Create a vocabulary (dict and reverse dictionary) based on music in a directory.
#     Input:
#     directory: where to get music from
#     Output:
#     word_dict: note to integer mapping
#     reverse_dictionary: integer to note mapping
#     mapped_data: mapping to int of every note
#     '''
#     data = MIDIModule.get_notes(directory, make_chords_from_notes)
#     print(data)
#     word_dict = MIDIModule.to_int(data)
#     print(word_dict)
#     print(len(word_dict))

#     reverse_dictionary = dict(zip(word_dict.values(),word_dict.keys()))
#     mapped_data = []
#     for element in data:
#         mapped_data.append(word_dict[element])
#     return word_dict, reverse_dictionary, mapped_data


# data_index = 0

# def generate_batch(batch_size, num_skips, skip_window, data):
#   global data_index
#   assert batch_size % num_skips == 0
#   assert num_skips <= 2 * skip_window

#   batch = np.ndarray(shape=(batch_size), dtype=np.int32)
#   labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

#   span = 2 * skip_window + 1  # [ skip_window target skip_window ]  Context that you predict

#   buffer = collections.deque(maxlen=span)

#   for _ in range(span):
#     buffer.append(data[data_index])
#     data_index = (data_index + 1) % len(data)

#   for i in range(batch_size // num_skips):
#     target = skip_window  # target label at the center of the buffer
#     targets_to_avoid = [skip_window]
#     for j in range(num_skips):
#       while target in targets_to_avoid:
#         target = random.randint(0, span - 1)
#       targets_to_avoid.append(target)
#       batch[i * num_skips + j] = buffer[skip_window]
#       labels[i * num_skips + j, 0] = buffer[target]
#     buffer.append(data[data_index])
#     data_index = (data_index + 1) % len(data)
#   # Backtrack a little bit to avoid skipping words in the end of a batch
#   data_index = (data_index + len(data) - span) % len(data)
#   return batch, labels


# # Step 4: Build and train a skip-gram model.
# def train_word2vec(directory, embedding_size = 256, num_steps =1000001 ):
#     dictionary, reverse_dictionary, data = build_vocabulary(directory)
#     vocabulary_size = len(dictionary)
#     batch_size = 8 #TODO:change this to 128 later
#     embedding_size = embedding_size  # Dimension of the embedding vector.
#     skip_window = 1  # How many words to consider left and right.
#     num_skips = skip_window * 2  # How many times to reuse an input to generate a label.
#     alpha = 0.01  # learning rate
#     num_steps = num_steps  # how long to train for

#     # We pick a random validation set to sample nearest neighbors. Here we limit the
#     # validation samples to the words that have a low numeric ID, which by
#     # construction are also the most frequent.
#     #TODO: scale these back up
#     valid_size = 2 # Random set of words to evaluate similarity on.
#     valid_window = 8  # Only pick dev samples in the head of the distribution.
#     valid_examples = np.random.choice(valid_window, valid_size, replace=False)
#     num_sampled = 16  # Number of negative examples to sample.

#     # Hand-picked validation set for music

#     #TODO: if we choose to use chords as validation, have to change them to string format to the mapping is preserved. The chords HAVE to be present in the data.

#     # fMajor = chord.Chord("F A C")
#     # cMajor = chord.Chord("C E G")
#     # gMajor = chord.Chord("G B D")
#     #
#     # valid_centers = [cMajor, fMajor, gMajor]  # 145: C-E-G, 545: F-A-C, 2180: G-B-D
#     # valid_neighbors = [[fMajor, gMajor, chord.Chord('A C E'), chord.Chord("G Bb Eb"), chord.Chord("F A D"), chord.Chord('G Bb D')],
#     #                    [cMajor, chord.Chord("Bb D F"), chord.Chord("D F A"),chord.Chord("C Eb Ab"), chord.Chord("Bb D G"),chord.Chord("C Eb G")],
#     #                    [chord.Chord("D F# A"),cMajor,chord.Chord("E G B"), chord.Chord("D F Bb"), chord.Chord("C E A"), chord.Chord("D F A")]] # V, IV, vi, IIIb, IIb, v




#     graph = tf.Graph()

#     with graph.as_default():
#         # Input data.
#         train_inputs = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])
#         train_labels = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, 1])
#         valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

#         # Ops and variables pinned to the CPU because of missing GPU implementation
#         # with tf.device('/cpu:0'):
#         # Look up embeddings for inputs.
#         embeddings = tf.Variable(
#             tf.random.uniform([vocabulary_size+1, embedding_size], -1.0, 1.0), name='embeddings')
#         embed = tf.nn.embedding_lookup(embeddings, train_inputs)

#         # Construct the variables for the NCE loss
#         nce_weights = tf.Variable(
#             tf.random.truncated_normal([vocabulary_size, embedding_size],
#                                 stddev=1.0 / math.sqrt(embedding_size)))
#         nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

#         # Compute the average NCE loss for the batch.
#         # tf.nce_loss automatically draws a new sample of the negative labels each
#         # time we evaluate the loss.
#         loss = tf.reduce_mean(
#             tf.nn.nce_loss(weights=nce_weights,
#                            biases=nce_biases,
#                            labels=train_labels,
#                            inputs=embed,
#                            num_sampled=num_sampled,
#                            num_classes=vocabulary_size))

#         # Construct an ADAm optimizer.
#         optimizer = tf.compat.v1.train.AdamOptimizer(alpha).minimize(loss)

#         # Compute the cosine similarity between minibatch examples and all embeddings.
#         norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True), name='norm')
#         normalized_embeddings = embeddings / norm
#         valid_embeddings = tf.nn.embedding_lookup(
#             normalized_embeddings, valid_dataset)
#         similarity = tf.matmul(
#             valid_embeddings, normalized_embeddings, transpose_b=True, name='similarity')

#         # Add variable initializer.
#         init = tf.compat.v1.global_variables_initializer()

#     # Step 5: Begin training.
#     with tf.compat.v1.Session(graph=graph) as session:
#         saver = tf.compat.v1.train.Saver()
#         # We must initialize all variables before we use them.
#         init.run()
#         print("Initialized")
#         average_loss = 0
#         results_C = []
#         results_F = []
#         results_G = []
#         for step in range(num_steps):
#             batch_inputs, batch_labels = generate_batch(
#                 batch_size, num_skips, skip_window, data)
#             feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

#             # We perform one update step by evaluating the optimizer op (including it
#             # in the list of returned values for session.run()
#             _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
#             average_loss += loss_val

#             if step % 2000 == 0:
#                 if step > 0:
#                     average_loss /= 2000
#                 # The average loss is an estimate of the loss over the last 2000 batches.
#                 print("Average loss at step ", step, ": ", average_loss)
#                 #print(average_loss)
#                 average_loss = 0

#             # Note that this is expensive (~20% slowdown if computed every 500 steps)
#             # get the 8  most closest to the validation set
#         #     if step % 10000 == 0:
#         #         sim = similarity.eval()
#         #         norm_embeddings = normalized_embeddings.eval()
#         #
#         #         # hand-picked validation examples
#         #         for i, center in enumerate(valid_centers):
#         #             center_i = dictionary[Hasher(center)]
#         #             sim_values = []
#         #             log_str = "Similarity to %s:" % dictionary[center]
#         #             for neighbor in valid_neighbors[i]:
#         #                 neighbor_i = dictionary[neighbor]
#         #                 center_embedding = norm_embeddings[center_i]
#         #                 neighbor_embedding = norm_embeddings[neighbor_i]
#         #                 cos_dist = distance.cosine(center_embedding, neighbor_embedding)
#         #                 sim_values.append(cos_dist)
#         #                 log_str = "%s %s:%s," % (log_str, dictionary[neighbor], cos_dist)
#         #             if i == 0:
#         #                 results_C.append(sim_values)
#         #             elif i == 1:
#         #                 results_F.append(sim_values)
#         #             else:
#         #                 results_G.append(sim_values)
#         #             print(log_str)
#         #
#         # final_embeddings = normalized_embeddings.eval()
#         # saver.save(session, "saves/word2vec_music_pc_train")
#         # np.savetxt('results_C.txt', results_C, delimiter=',', newline='\n')
#         # np.savetxt('results_F.txt', results_F, delimiter=',', newline='\n')
#         # np.savetxt('results_G.txt', results_G, delimiter=',', newline='\n')
#         return normalized_embeddings



# def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
#   assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
#   plt.figure(figsize=(18, 18))  # in inches
#   for i, label in enumerate(labels):
#     x, y = low_dim_embs[i, :]
#     plt.scatter(x, y)
#     plt.annotate(label,
#                  xy=(x, y),
#                  xytext=(5, 2),
#                  textcoords='offset points',
#                  ha='right',
#                  va='bottom')

#   plt.savefig(filename)


# def train_embedding(directory, embedding_size=256, num_steps = 1):
#     return train_word2vec(directory, embedding_size)

# def get_embedding(elements, trained_embeddings = None, embedding_size=256):
#     """Returns Embeddings of elements.

#     One-hot for notes. If the get_notes methods are used with
#     make_notes_from_chords = True then it is is multi-hot for notes,
#     else it is one hot

#     Args:
#         elements    : List of elements (notes, chords, durations).
#         graph : a trained embedding layer
#         embedding_size: dimesion of embedding.
#     Returns:
#         embeddings  : positional encoding + trained embedding.
#     """
#     dict = MIDIModule.to_int(elements)
#     embeddings = tf.nn.embedding_lookup(params = trained_embeddings, ids = tf.convert_to_tensor(list(map(lambda x: dict[x], elements))))
#     for i in range(len(elements)):
#         embeddings[i] += positional_encoding(i, embedding_size)
#     return embeddings

# def get_sequences(directory, sequence_length, embedding_size=256,num_steps = 1000001 ):
#     """Retrieve the input and output sequences.

#     Args:
#         directory          : Where to get notes.
#         sequence_length: dimension of embedding.
#         sequence_length : Length of the sequence ( ex. len(note_list) ).

#     Returns:
#         network_input   : Input sequence.
#         network_output  : Output sequence.
#     """

#     embeddings = train_word2vec(directory, embedding_size, num_steps)
#     notes = MIDIModule.get_notes(directory)
#     notes = get_embedding(notes, embeddings, embedding_size)

#     network_input = []
#     network_output = []
#     for i in range(0, len(notes) - sequence_length, 1):
#         sequence_in = notes[i:i + sequence_length]
#         sequence_out = notes[i + sequence_length]
#         network_input.append(sequence_in)
#         network_output.append(sequence_out)

#     return network_input, network_output