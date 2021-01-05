# Local Imports
# ...

# Third-party imports
from music21 import *
import numpy as np

class word2vec():
    """...

    ...
    """

    def __init__(self, n=10, alpha=1e-4, window_size=2):
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
            # context = list()
            # for j in range(i - self.window_size, i + self.window_size + 1):
            #     if ((j != i) and (j <= (len(data) - 1)) and (j >= 0)):
            #         context.append(self.to_nhot(data[j]))
                    
            processed_data.append([target])
            
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

    # def train(self, training_data):
    #     """...

    #     ...

    #     Args:
    #         training_data

    #     Returns:
    #         ...
    #     """

    #     self.w1 = np.random.uniform(-0.8, 0.8, (self.n_notes, self.n))     # embedding matrix
    #     self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.n_notes))     # context matrix
        
    #     for i in range(0, self.epochs):

    #         self.loss = 0

    #         for n_t, n_c in training_data:

    #             y_pred, h, u = self.forward_prop(n_t)

    #             EI = np.sum([np.subtract(y_pred, note) for note in n_c], axis=0)

    #             self. back_prop(EI, h, n_t)

    #             self.loss += -np.sum([u[note.index(1)] for note in n_c]) + len(n_c) * np.log(np.sum(np.exp(u)))
    #             #self.loss += -2*np.log(len(n_c)) -np.sum([u[word.index(1)] for word in n_c]) + (len(n_c) * np.log(np.sum(np.exp(u))))
                
    #         print('EPOCH:',i, 'LOSS:', self.loss)
    #     pass

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
