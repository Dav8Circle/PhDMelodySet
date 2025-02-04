"""
Module for computing corpus-based features from melodic n-grams, similar to FANTASTIC's
implementation. This module handles the corpus analysis and saves statistics to JSON.
The actual feature calculations are handled in features.py.
"""
from collections import Counter, defaultdict
import json
from typing import List, Dict, Tuple
from mtypes import FantasticTokenizer
from representations import Melody, read_midijson

def _convert_tuples_to_strings(data: Dict) -> Dict:
    """Convert tuple keys to strings for JSON serialization.
    
    Parameters
    ----------
    data : Dict
        Dictionary with potential tuple keys
        
    Returns
    -------
    Dict
        Dictionary with tuple keys converted to strings
    """
    converted = {}
    for key, value in data.items():
        if isinstance(key, tuple):
            # Convert tuple to string representation 
            # json.dumps converts None to "null"
            key_list = tuple("None" if x is None else x for x in key)
            str_key = str(key_list)  # Use str() to preserve tuple format
            converted[str_key] = value
        else:
            converted[key] = value
    return converted

def _convert_strings_to_tuples(key: str) -> Tuple:
    """Convert a string-encoded tuple key back to a tuple.
    
    Parameters
    ----------
    key : str
        String-encoded tuple key
        
    Returns
    -------
    Tuple
        Tuple converted from string
    """
    try:
        # Remove parentheses and split on comma
        key_str = key.strip('()').split(',')
        # Convert "None" strings back to None and strip quotes/spaces
        tuple_key = tuple(None if x.strip().strip("'\"") == "None" else x.strip().strip("'\"") for x in key_str)
        return tuple_key
    except (AttributeError, ValueError):
        # If not a tuple string, return the original key
        return key

def compute_corpus_ngrams(melodies: List[Melody], n_range: Tuple[int, int] = (1, 4)) -> Dict:
    """Compute n-gram frequencies across the entire corpus.
    
    Parameters
    ----------
    melodies : List[Melody]
        List of Melody objects to analyze
    n_range : Tuple[int, int]
        Range of n-gram lengths to consider (min, max)
        
    Returns
    -------
    Dict
        Dictionary containing corpus-wide n-gram statistics
    """
    tokenizer = FantasticTokenizer()
    corpus_counts = defaultdict(Counter)
    
    for melody in melodies:
        # Tokenize melody
        tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)
        
        # Get n-grams for each length
        for n in range(n_range[0], n_range[1] + 1):
            counts = tokenizer.ngram_counts(n)
            
            # Update corpus-wide counts
            corpus_counts[n].update(counts)
    
    # Convert defaultdict and Counter objects to regular dicts for JSON serialization
    frequencies = {'document_frequencies': {}}
    for n, counts in corpus_counts.items():
        for k, v in counts.items():
            frequencies['document_frequencies'][str(k)] = {'count': v}
    
    return {
        'document_frequencies': frequencies['document_frequencies'],
        'corpus_size': len(melodies),
        'n_range': n_range
    }

def save_corpus_stats(stats: Dict, filename: str) -> None:
    """Save corpus statistics to a JSON file.
    
    Parameters
    ----------
    stats : Dict
        Corpus statistics from compute_corpus_ngrams
    filename : str
        Path to save JSON file
    """
    with open(filename, 'w') as f:
        json.dump(stats, f, indent=2)

def load_corpus_stats(filename: str) -> Dict:
    """Load corpus statistics from a JSON file.
    
    Parameters
    ----------
    filename : str
        Path to JSON file
        
    Returns
    -------
    Dict
        Corpus statistics dictionary
    """
    with open(filename, encoding='utf-8') as f:
        stats = json.load(f)
    
    # Convert string keys back to tuples where needed
    stats['document_frequencies'] = {
        _convert_strings_to_tuples(k): v for k, v in stats['document_frequencies'].items()
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    melodies = []
    filename = '/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/corpus_stats2.json'
    for i in range(0, 1920, 1):
        melody_data = read_midijson(
            '/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/mididata5.json')[i]
        melody = Melody(melody_data, tempo=100)
        melodies.append(melody)
    
    # Compute and save corpus statistics
    corpus_stats = compute_corpus_ngrams(melodies)
    save_corpus_stats(corpus_stats, 'corpus_stats2.json')
    
    # Load and verify
    loaded_stats = load_corpus_stats('corpus_stats2.json')
    print("Corpus statistics saved and loaded successfully.")
    print(f"Corpus size: {loaded_stats['corpus_size']} melodies")
    print(f"N-gram lengths: {loaded_stats['n_range']}")
