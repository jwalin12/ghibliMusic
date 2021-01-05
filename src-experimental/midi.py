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
    def list_instruments(midi):
        """...
        
        Args:
            midi    : ...

        Returns:
            None
        """

        partStream = midi.parts.stream()
        print("List of instruments found on MIDI file:")
        for p in partStream:
            print (p.partName)

    @staticmethod
    def get_analysis(midi):
        """...

        Args:
            midi    : ...
        
        Returns:
            None
        """

        timeSignature = midi.getTimeSignatures()[0]
        music_analysis = midi.analyze('key')

        print("Music time signature: {0}/{1}".format(timeSignature.beatCount, timeSignature.denominator))
        print("Expected music key: {0}".format(music_analysis))
        print("Music key confidence: {0}".format(music_analysis.correlationCoefficient))

        print("Other music key alternatives:")
        for analysis in music_analysis.alternateInterpretations:
            if (analysis.correlationCoefficient > 0.5):
                print(analysis)

    @staticmethod
    def get_notes(directory):
        """Converts MIDI files within a directory into a single note array.

        Directory is a string of the filepath from which to import the midi
        files.
        If chord appends elements seperated by '.' sequential order is preserved.
        Output array is a list of all elements in sequential order.

        Args:
            directory   : Path where the MIDI file(s) are located.
            make_chords_from_notes: Boolean that indicates whether or not chords should
            be treated as seperate objects or collections of elements.

        Return:
            elements   : List of elements transcribed from the MIDI file(s).
        """

        notes = []
        for file in glob.glob(directory + '/*.mid'):
            midi = converter.parse(file)
            notes_to_parse = None

            parts = instrument.partitionByInstrument(midi)

            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.elements

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n.pitch) for n in element.notes))

        return notes

    @staticmethod
    def to_int(notes):
        """Converts elements and chords into integer values.
        Args:
            notes   : List of elements.
        Returns:
            note_to_int : Dictionary where the KV-pairs are a note and number,
                            respectively.
        """

        pitchnames = set(item for item in notes)
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

        return note_to_int



