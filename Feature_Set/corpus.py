"""
Module for computing corpus-based features from melodic n-grams, similar to FANTASTIC's
implementation. This module handles the corpus analysis and saves statistics to JSON.
The actual feature calculations are handled in features.py.
"""
from collections import Counter, defaultdict
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import multiprocessing as mp
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

def process_melody_ngrams(args) -> Counter:
    """Process n-grams for a single melody.
    
    Parameters
    ----------
    args : tuple
        Tuple containing (melody, n_range)
        
    Returns
    -------
    Counter
        Counter object containing n-gram counts for the melody
    """
    melody, n_range = args
    tokenizer = FantasticTokenizer()
    tokens = tokenizer.tokenize_melody(melody.pitches, melody.starts, melody.ends)
    
    counts = Counter()
    for n in range(n_range[0], n_range[1] + 1):
        counts.update(tokenizer.ngram_counts(n))
    return counts

def compute_corpus_ngrams(melodies: List[Melody], n_range: Tuple[int, int] = (1, 4)) -> Dict:
    """Compute n-gram frequencies across the entire corpus using multiprocessing.
    
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
    # Prepare arguments for multiprocessing
    args = [(melody, n_range) for melody in melodies]
    
    # Use all available CPU cores
    num_cores = mp.cpu_count()
    
    # Create a pool of workers
    with mp.Pool(num_cores) as pool:
        # Process melodies in parallel with progress bar
        results = list(tqdm(
            pool.imap(process_melody_ngrams, args),
            total=len(melodies),
            desc=f"Computing n-grams using {num_cores} cores"
        ))
    
    # Combine results from all processes
    total_counts = Counter()
    for counts in results:
        total_counts.update(counts)
    
    # Format results for JSON serialization
    frequencies = {'document_frequencies': {}}
    for k, v in total_counts.items():
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

def load_melody(idx: int, filename: str) -> Melody:
    """Load a single melody from a JSON file.
    
    Parameters
    ----------
    idx : int
        Index of melody to load
    filename : str
        Path to JSON file
        
    Returns
    -------
    Melody
        Loaded melody object
    """
    melody_data = read_midijson(filename)
    if idx >= len(melody_data):
        raise IndexError(f"Index {idx} is out of range for file with {len(melody_data)} melodies")
    return Melody(melody_data[idx], tempo=100)

if __name__ == "__main__":
    # Example usage
    filename = '/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/miq_midi.json'
    
    # Get the actual number of melodies in the file
    melody_data = read_midijson(filename)
    num_melodies = len(melody_data)
    print(f"Found {num_melodies} melodies in file")
    
    # Create arguments for parallel loading
    num_cores = mp.cpu_count()
    melody_indices = [(i, filename) for i in range(num_melodies)]
    
    # Load melodies in parallel
    with mp.Pool(num_cores) as pool:
        melodies = list(tqdm(
            pool.starmap(load_melody, melody_indices),
            total=len(melody_indices),
            desc=f"Loading melodies using {num_cores} cores"
        ))

    # Compute and save corpus statistics
    corpus_stats = compute_corpus_ngrams(melodies)
    save_corpus_stats(corpus_stats, 'original_mel_miq_corpus_stats.json')

    # Load and verify
    loaded_stats = load_corpus_stats('original_mel_miq_corpus_stats.json')
    print("Corpus statistics saved and loaded successfully.")
    print(f"Corpus size: {loaded_stats['corpus_size']} melodies")
    print(f"N-gram lengths: {loaded_stats['n_range']}")
