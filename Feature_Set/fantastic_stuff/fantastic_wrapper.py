import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri

pandas2ri.activate()

r.source("Feature_Set/fantastic_stuff/Feature_Value_Summary_Statistics.R")
r.source("Feature_Set/fantastic_stuff/Frequencies_Summary_Statistics.R")
r.source("Feature_Set/fantastic_stuff/Feature_Similarity.R")
r.source("Feature_Set/fantastic_stuff/M-Type_Corpus_Features.R")
r.source("Feature_Set/fantastic_stuff/M-Type_Summary_Statistics.R")
r.source("Feature_Set/fantastic_stuff/Fantastic.R")

def compute_features_in_r(melody):
    # Convert melody data to an R data frame or list as needed
    r_melody_data = ro.vectors.ListVector(melody)
    
    # Debug: Print the R data to check its structure
    print("R melody data:", r_melody_data)
    
    # Call the R function
    r_compute_features = ro.r['compute.features']  # Ensure this is the correct function name
    try:
        result = r_compute_features(r_melody_data)
    except Exception as e:
        print("Error calling R function:", e)
        raise
    
    # Convert the result back to a Python object if needed
    return result
if __name__ == "__main__":
    # Example melody data
    example_melody = {
        'pitches': [60, 62, 64, 65, 67, 69, 71, 72],
        'starts': [0, 1, 2, 3, 4, 5, 6, 7],
        'ends': [1, 2, 3, 4, 5, 6, 7, 8],
        'durs': [1, 1, 1, 1, 1, 1, 1, 1],
        'dur16': [4, 4, 4, 4, 4, 4, 4, 4]
    }

    # Compute features using the R function
    features = compute_features_in_r(example_melody)
    
    # Print the computed features
    print(features)

