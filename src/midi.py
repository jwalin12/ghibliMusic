""" The class for dealing with MIDI data.

...

Used for:
  Importing
  Cleaning
  Parsing
  Encoding
  ...
"""

from music21 import converter, instrument, note, chord
import glob



'''directory is a string of directory from which to get midi files.
If chord appends notes seperated by.Sequential order is preserved. output array is a list of all notes in
sequential order.'''
def get_notes(directory):
    notes = []
    for file in glob.glob(directory+"/*.mid"):
        midi = converter.parse(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts: # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes


'''directory is a string of directory from which to get midi files.
If chord appends notes separated by.Sequential order is preserved. output array is a list of lists, where each entry
 is a list of all notes in a particular song in sequential order, '''

def get_notes_by_song(directory):
    all_notes = []
    for file in glob.glob(directory+"/*.mid"):
        notes = []
        midi = converter.parse(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts: # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
        all_notes.append(notes)
    return all_notes

'''converts note to ints. These ints can then be used to create n-hot arrays. Chords have their own int'''
def get_note_to_int(notes):
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    return note_to_int



'''gets input and output sequences of sequence length.'''

def get_sequences(notes, sequence_length):
    note_to_int = get_note_to_int(notes)
    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    return network_input, network_output
