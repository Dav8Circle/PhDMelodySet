import csv
import json
import pretty_midi

def get_original_sequence(file):
    midi_data = pretty_midi.PrettyMIDI(file).instruments[0].notes
    midi_data.pop()  # Drop the last note
    return str(midi_data)[1:-1]

with open("item-bank.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    output_file = open("original_mel_miq_midi.json", "a", encoding='utf-8')
    output_file.write("[\n")
    count = 1

    for row in reader:
        name = str(row["original_melody"])
        file = "mid/{melody}.mid".format(melody=name)

        output = get_original_sequence(file)

        output_string = {"ID": count,
                         "Original Melody": name,
                         "MIDI Sequence": output,
                         }
        output_dict = json.dumps(output_string, indent=4)
        output_file.write(output_dict)
        output_file.write(",\n")
        count += 1

    output_file.write("]")
    output_file.close()
