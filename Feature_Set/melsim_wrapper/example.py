from .melsim import get_similarity_from_midi, install_dependencies

# Install dependencies
install_dependencies()

appenzel_path = "Feature_Set/melsim_wrapper/appenzel.mid"
arabic_path = "Feature_Set/melsim_wrapper/arabic01.mid"

# Calculate similarity between two MIDI files
similarity_value = get_similarity_from_midi(
    appenzel_path,
    arabic_path,
    method="Jaccard",           # Using Jaccard similarity measure
    transformation="pitch"      # Compare raw pitch values
)
print(f"Jaccard pitch similarity: {similarity_value:.3f}")

# Try another combination
similarity_value = get_similarity_from_midi(
    appenzel_path,
    arabic_path,
    method="edit_sim",         # Using edit distance similarity
    transformation="parsons"   # Compare melodic contours
)
print(f"Edit distance similarity using Parsons code: {similarity_value:.3f}")
