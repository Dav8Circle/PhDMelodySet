import csv
import json
import pretty_midi



class OriginalMelody:

    def __init__(self, file):
        self.file = file
        self.midi_data = pretty_midi.PrettyMIDI(file).instruments[0].notes

    def transpose(self, amount):
        for note in self.midi_data:
            note.pitch += amount
        return self.midi_data


class ModifiedMelody(OriginalMelody):

    def __init__(self, in_key, contour_dif, change_note, displacement, oddity, discrimination, difficulty):
        super().__init__(file)
        self.in_key = in_key
        self.contour_dif = contour_dif
        # change note is 1-indexed
        self.change_note = change_note - 1
        self.displacement = displacement
        self.oddity = oddity
        self.discrimination = discrimination
        self.difficulty = difficulty

    def directionFinder(self):
        contour_scalar = self.midi_data[(self.change_note - 2)].pitch - self.midi_data[(self.change_note - 1)].pitch
        return contour_scalar

    def changeMelody(self, amount):
        if self.directionFinder() > 0 & contour_dif == 0:
            self.midi_data[self.change_note].pitch -= displacement

        elif self.directionFinder() > 0 & contour_dif == 4:
            self.midi_data[self.change_note].pitch += displacement

        elif self.directionFinder() <= 0 & contour_dif == 0:
            self.midi_data[self.change_note].pitch += displacement

        elif self.directionFinder() <= 0 & contour_dif == 4:
            self.midi_data[self.change_note].pitch -= displacement

        self.transpose(amount)
        return self.midi_data


def create_modified_sequence(file, in_key, contour_dif, change_note, displacement, oddity, discrimination,
                    difficulty):
    modified = ModifiedMelody(in_key, contour_dif, change_note, displacement, oddity, discrimination, difficulty)
    modified.midi_data.pop()
    
    if oddity == 1:
        return modified.changeMelody(0)
    elif oddity == 2:
        return modified.changeMelody(1)
    elif oddity == 3:
        return modified.changeMelody(2)


with open("item-bank.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    output_file = open("miq_midi.json", "a", encoding='utf-8')
    output_file.write("[\n")
    count = 1

    for row in reader:
        name = str(row["original_melody"])
        in_key = str(row["in_key"])
        contour_dif = int(row["contour_dif"])
        change_note = int(row["change_note"])
        displacement = int(row["displacement"])
        oddity = int(row["oddity"])
        discrimination = float(row["discrimination"])
        difficulty = float(row["difficulty"])

        file = "mid/{melody}.mid".format(melody=name)

        output = str(create_modified_sequence(file, in_key, contour_dif, change_note, displacement, oddity, discrimination,
                                     difficulty))[1:-1]

        output_string = {"ID": count,
                         "Original Melody": name,
                         "In Key": in_key,
                         "Contour Dif": contour_dif,
                         "Change Note": change_note,
                         "Displacement": displacement,
                         "Oddity": oddity,
                         "Discrimination": discrimination,
                         "Difficulty": difficulty,
                         "MIDI Sequence": output,
                         }
        output_dict = json.dumps(output_string, indent=4)
        output_file.write(output_dict)
        output_file.write(",\n")
        count += 1

    output_file.write("]")
    output_file.close()
