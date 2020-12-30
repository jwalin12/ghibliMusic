# Local imports

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
    def to_int(notes):
        """Converts notes and chords into integer values.

        Args:
            notes   : List of notes.

        Returns:
            note_to_int : Dictionary where the KV-pairs are a note and number,
                            respectively.
        """

        pitchnames = set(item for item in notes)
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

        return note_to_int

    @staticmethod
    def get_notes(directory,  make_chords_from_notes = False):
        """Converts MIDI files within a directory into a single note array.

        Directory is a string of the filepath from which to import the midi
        files.
        If chord appends notes seperated by '.' sequential order is preserved. 
        Output array is a list of all notes in sequential order.

        Args:
            directory   : Path where the MIDI file(s) are located.
            make_chords_from_notes: Boolean that indicates whether or not chords should
            be treated as seperate objects or collections of notes.

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
                    if (make_chords_from_notes):
                        notes.append("CHORD")
                        for n in element.notes:
                            notes.append(str(n.pitch))
                        notes.append(str(element.duration))
                    else:
                        el = notes.append('.'.join(str(n) for n in element.notes))
                        notes.append(el)
                        notes.append(str(element.duration))

        return notes

    @staticmethod
    def get_notes_by_song(directory, make_chords_from_notes = False):
        """Converts MIDI files within a directory into a multidimensional
        array where the row is the song and the column is the note.

        Directory is a string of the filepath from which to import the midi
        files.
        If chord appends notes seperated by '.' sequential order is preserved. 
        Output array is a list of all notes in sequential order.

        Args:
            directory   : Path where the MIDI file(s) are located.
            make_chords_from_notes: indicates whether chords should be treated
            as seperate objects or collections of notes.

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
                    notes.append(str(element.duration))

                elif isinstance(element, chord.Chord):
                    if (make_chords_from_notes):
                        notes.append("CHORD")
                        for n in element.notes:
                            notes.append(str(n.pitch))
                        notes.append(str(element.duration))
                    else:
                        el = notes.append('.'.join(str(n) for n in element.notes))
                        notes.append(el)
                        notes.append(str(element.duration))

            songs.append(notes)
            
        return songs




