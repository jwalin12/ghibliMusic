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



'''directory is a string of directory from which to get midi files. returns a list of all notes played. 
If chord appends notes seperated by . Sequential order is preserved. '''
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

'''converts note to ints. These ints can then be used to create n-hot arrays. Chords have their own int'''
def get_note_to_int(notes):
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    return note_to_int
