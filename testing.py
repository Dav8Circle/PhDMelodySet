from Feature_Set.features import get_all_features_csv

# must use if __name__ == "__main__" guard to avoid circular import
if __name__ == "__main__":
    get_all_features_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/Essen_Analysis/essen_midi_sequences.json", "temp", "/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/Essen_Analysis/essen_corpus_stats.json")
