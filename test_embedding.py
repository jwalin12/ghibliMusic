from src.word2vec import word2vec_music, get_embedding
from src.midi import MIDIModule
import tensorflow as tf
directory = '/Users/jwalinjoshi/ghibliMusic/'
graph = tf.Graph()
layer = word2vec_music(graph)
layer.build(directory, num_steps=20000)
notes = MIDIModule.get_notes(directory)
input = MIDIModule.get_encododed_notes(directory)


print(get_embedding(elements=input,trained_embeddings=layer.embeddings, graph = graph))
