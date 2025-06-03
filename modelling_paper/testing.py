from Feature_Set.features import get_all_features
from Feature_Set.corpus import make_corpus_stats
# must use if __name__ == "__main__" guard to avoid circular import
if __name__ == "__main__":
    make_corpus_stats("/Users/davidwhyatt/Downloads/01_Essen Folksong Database (.mid-conversions)", "testing_corpus_stats")
    get_all_features("/Users/davidwhyatt/Downloads/01_Essen Folksong Database (.mid-conversions)", "testing_test", "testing_corpus_stats.json")
    # get_all_features("/Users/davidwhyatt/Downloads/01_Essen Folksong Database (.mid-conversions)", "test2.csv")
