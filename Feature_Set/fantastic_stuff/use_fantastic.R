source("path/to/Fantastic.R")
results <- compute.features(melody.filenames = list.files(path = "/Users/davidwhyatt/Documents/mid/DB", pattern = ".csv"), dir = "your/melody/directory", output = "melody.wise", use.segmentation = TRUE)
print(results)
