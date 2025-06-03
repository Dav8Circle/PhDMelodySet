from Feature_Set.features import get_all_features

# must use if __name__ == "__main__" guard to avoid circular import
if __name__ == "__main__":
    get_all_features("/Users/davidwhyatt/Downloads/01_Essen Folksong Database (.mid-conversions)", "testing_test.csv", "/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/modelling_paper/essen_corpus_stats.json")
    # get_all_features("/Users/davidwhyatt/Downloads/01_Essen Folksong Database (.mid-conversions)", "test2.csv")
