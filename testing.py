from Feature_Set.features import get_all_features_csv

# must use if __name__ == "__main__" guard to avoid circular import
if __name__ == "__main__":
    get_all_features_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/original_mel_miq_midi.json", "original_mel_miq_mels", "/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/original_mel_miq_corpus_stats.json")
