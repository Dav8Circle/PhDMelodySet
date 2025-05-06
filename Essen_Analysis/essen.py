from Feature_Set.features import get_all_features_csv

if __name__ == "__main__":
    get_all_features_csv(
        input_path='/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/Essen_Analysis/essen_midi_sequences.json',
        output_path='essen_features',
        corpus_path='/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/Essen_Analysis/essen_corpus_stats.json'
    )