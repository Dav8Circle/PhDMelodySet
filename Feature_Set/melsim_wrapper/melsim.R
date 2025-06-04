#!/usr/bin/env Rscript

# Load required packages
if (!require("memoise")) {
    install.packages("memoise")
    library(memoise)
}

# Create memoized functions for better caching
get_melody <- memoise(function(pitches, starts, ends) {
    melody_factory$new(
        mel_data = tibble::tibble(
            onset = as.numeric(starts),
            pitch = as.numeric(pitches),
            duration = as.numeric(ends) - as.numeric(starts)
        )
    )
})

get_sim_measure <- memoise(function(method, transformation) {
    sim_measure_factory$new(
        name = method,
        full_name = method,
        transformation = transformation,
        parameters = list(),
        sim_measure = method
    )
})

# Function to run melsim with given melody attributes
run_melsim <- function(melody1_pitches, melody1_starts, melody1_ends,
                      melody2_pitches, melody2_starts, melody2_ends,
                      method = "Jaccard", transformation = "pitch") {
    
    if (!require("melsim")) {
        stop("melsim package not found. Please install it first.")
    }
    
    # Get or create melody objects (now using memoized functions)
    melody1 <- get_melody(melody1_pitches, melody1_starts, melody1_ends)
    melody2 <- get_melody(melody2_pitches, melody2_starts, melody2_ends)
    
    # Get or create similarity measure (now using memoized function)
    sim_measure <- get_sim_measure(method, transformation)
    
    # Run melsim
    result <- melody1$similarity(melody2, sim_measure)$sim
    
    # Return result as JSON
    cat(jsonlite::toJSON(result))
}

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Parse arguments
if (length(args) < 6) {
    stop("Usage: Rscript melsim.R melody1_pitches melody1_starts melody1_ends melody2_pitches melody2_starts melody2_ends [method] [transformation]")
}

# Convert string arguments to vectors (vectorized operation)
melody1_pitches <- as.numeric(strsplit(args[1], ",")[[1]])
melody1_starts <- as.numeric(strsplit(args[2], ",")[[1]])
melody1_ends <- as.numeric(strsplit(args[3], ",")[[1]])
melody2_pitches <- as.numeric(strsplit(args[4], ",")[[1]])
melody2_starts <- as.numeric(strsplit(args[5], ",")[[1]])
melody2_ends <- as.numeric(strsplit(args[6], ",")[[1]])

# Get optional arguments
method <- if (length(args) > 6) args[7] else "Jaccard"
transformation <- if (length(args) > 7) args[8] else "pitch"

# Run melsim
run_melsim(
    melody1_pitches, melody1_starts, melody1_ends,
    melody2_pitches, melody2_starts, melody2_ends,
    method, transformation
) 