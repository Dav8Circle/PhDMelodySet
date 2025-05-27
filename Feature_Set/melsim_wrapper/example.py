import numpy as np
from melsim import get_similarity, install_dependencies

# Install dependencies
install_dependencies()

# Example melodies represented as MIDI pitch numbers
melody1_pitches = np.array([60, 62, 64, 65, 67])  # C4, D4, E4, F4, G4
melody1_starts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # One note per second
melody1_ends = np.array([0.9, 1.9, 2.9, 3.9, 4.9])    # Each note 0.9s long

melody2_pitches = np.array([60, 62, 64, 62, 60])  # C4, D4, E4, D4, C4 
melody2_starts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
melody2_ends = np.array([0.9, 1.9, 2.9, 3.9, 4.9])

# Calculate similarity using different methods and transformations
similarity_value = get_similarity(
    melody1_pitches, melody1_starts, melody1_ends,
    melody2_pitches, melody2_starts, melody2_ends,
    method="Jaccard",           # Using Jaccard similarity measure
    transformation="pitch"      # Compare raw pitch values
)
print(f"Jaccard pitch similarity: {similarity_value:.3f}")

# Try another combination
similarity_value = get_similarity(
    melody1_pitches, melody1_starts, melody1_ends,
    melody2_pitches, melody2_starts, melody2_ends,
    method="edit_sim",         # Using edit distance similarity
    transformation="parsons"   # Compare melodic contours
)
print(f"Edit distance similarity using Parsons code: {similarity_value:.3f}")
