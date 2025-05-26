from Feature_Set.features import get_all_features

# must use if __name__ == "__main__" guard to avoid circular import
if __name__ == "__main__":
    get_all_features("/Users/davidwhyatt/Downloads/01_Essen Folksong Database (.mid-conversions)", "test.csv", "/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/essen_corpus_stats.json")
