from src.word2vec import get_sequences

directory = '/Users/jwalinjoshi/ghibliMusic/'
print(get_sequences(directory, 4, num_steps = 20000))