# Local imports
# ...

# Third-party imports
from music21 import converter, instrument, note, chord
import glob

class MIDIModule():
    """ The class for dealing with MIDI data.

    WIP: The class provides statics methods.

    Source:
    Towards Data Science - How to Generate Music using a LSTM Neural Network
    in Keras by Sigurour Skuli

    Used for:
      Importing
      Cleaning
      Parsing
      Encoding
      ...
    """

    @staticmethod
    def get_notes(directory):
        """Converts MIDI files within a directory into a single note array.

        Directory is a string of the filepath from which to import the midi
        files.
        If chord appends notes seperated by '.' sequential order is preserved. 
        Output array is a list of all notes in sequential order.

        Args:
            directory   : Path where the MIDI file(s) are located.

        Return:
            notes   : List of notes transcribed from the MIDI file(s).
        """

        notes = []
        for file in glob.glob(directory+"/*.mid"):
            midi = converter.parse(file)
            notes_to_parse = None

            parts = instrument.partitionByInstrument(midi)

            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

        return notes


    @staticmethod
    def get_notes_by_song(directory):
        """Converts MIDI files within a directory into a multidimensional
        array where the row is the song and the column is the note.

        Directory is a string of the filepath from which to import the midi
        files.
        If chord appends notes seperated by '.' sequential order is preserved. 
        Output array is a list of all notes in sequential order.

        Args:
            directory   : Path where the MIDI file(s) are located.

        Return:
            songs   : A two-dimensional list.
        """

        songs = []
        for file in glob.glob(directory+"/*.mid"):
            notes = []
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)

            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

            songs.append(notes)
            
        return songs


    @staticmethod
    def to_int(notes):
        """Converts notes and chords into integer values.

        Args:
            notes   : List of notes.

        Returns:
            note_to_int : Dictionary where the KV-pairs are a note and number,
                            respectively.
        """

        pitchnames = sorted(set(item for item in notes))
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

        return note_to_int


    @staticmethod
    def get_sequences(notes, sequence_length):
        """Retrieve the input and output sequences.

        Args:
            notes           : List of notes.
            sequence_length : Length of the sequence ( ex. len(note_list) ).

        Returns:
            network_input   : Input sequence.
            network_output  : Output sequence.
        """

        note_to_int = MIDIModule.to_int(notes)

        network_input = []
        network_output = []

        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]

            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

        return network_input, network_output
        