import json

class Melody:
    """Class to represent a melody from a MIDI sequence. This class is used to extract
    information from a json file containing MIDI sequence data, formatted according to
    pretty_midi conventions:
    A note is represented as a string in the format:
    'Note(start=0.0, end=0.25, pitch=60, velocity=100)'
    We don't need the velocity, so we can ignore it here.
    
    Attributes:
        pitches (list[int]): List of MIDI note pitches in order of appearance
        starts (list[float]): List of note start times in order of appearance
        ends (list[float]): List of note end times in order of appearance
    """
    def __init__(self, midi_data: dict):
        """Initialize a Melody object from MIDI sequence data.
        
        Args:
            midi_data (dict): Dictionary containing MIDI sequence data
        """
        self._midi_data = midi_data
        self._midi_sequence = midi_data['MIDI Sequence'].split('), ')

    @property
    def pitches(self) -> list[int]:
        """Extract pitch values from MIDI sequence.
        
        Returns:
            list[int]: List of MIDI pitch values in order of appearance
        """
        pitches = []
        for note in self._midi_sequence:
            pitch_start = note.find('pitch=') + 6
            pitch_end = note.find(',', pitch_start)
            if pitch_end == -1:  # Handle the last note which ends with ')'
                pitch_end = note.find(')', pitch_start)
            pitch = int(note[pitch_start:pitch_end])
            pitches.append(pitch)
        return pitches

    @property
    def starts(self) -> list[float]:
        """Extract start times from MIDI sequence.
        
        Returns:
            list[float]: List of MIDI note start times in order of appearance
        """
        starts = []
        for note in self._midi_sequence:
            start_start = note.find('start=') + 6
            start_end = note.find(',', start_start)
            start = float(note[start_start:start_end])
            starts.append(start)
        return starts

    @property
    def ends(self) -> list[float]:
        """Extract end times from MIDI sequence.
        
        Returns:
            list[float]: List of MIDI note end times in order of appearance
        """
        ends = []
        for note in self._midi_sequence:
            end_start = note.find('end=') + 4
            end_end = note.find(',', end_start)
            end = float(note[end_start:end_end])
            ends.append(end)
        return ends

def read_midijson(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Example usage
melody_data = read_midijson('/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/mididata5.json')[0]
melody = Melody(melody_data)
print(melody.pitches)
print(melody.starts)
print(melody.ends)
