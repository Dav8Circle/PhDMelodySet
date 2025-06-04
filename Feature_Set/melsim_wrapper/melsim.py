"""
This is a Python wrapper for the R package 'melsim' (https://github.com/sebsilas/melsim).
This wrapper allows the user to easily interface with the melsim package using numpy arrays
representing melodies.

Melsim is a package for computing similarity between melodies, and is being developed by
Sebastian Silas (https://sebsilas.com/) and Klaus Frieler
(https://www.aesthetics.mpg.de/en/the-institute/people/klaus-frieler.html).

Melsim is based on SIMILE, which was written by Daniel Müllensiefen and Klaus Frieler in 2003/2004.
This package is used to compare two or more melodies pairwise across a range of similarity measures.
Not all similarity measures are implemented in melsim, but the ones that are can be used here.

All of the following similarity measures are implemented and functional in melsim:
Please be aware that the names of the similarity measures are case-sensitive.

Num:        Name:
1           Jaccard
2       Kulczynski2
3            Russel
4             Faith
5          Tanimoto
6              Dice
7            Mozley
8            Ochiai
9            Simpson
10           cosine
11          angular
12      correlation
13        Tschuprow
14           Cramer
15            Gower
16        Euclidean
17        Manhattan
18         supremum
19         Canberra
20            Chord
21         Geodesic
22             Bray
23          Soergel
24           Podani
25        Whittaker
26         eJaccard
27            eDice
28   Bhjattacharyya
29       divergence
30        Hellinger
31    edit_sim_utf8
32         edit_sim
33      Levenshtein
34          sim_NCD
35            const
36          sim_dtw

The following similarity measures are not currently functional in melsim:
1    count_distinct (set-based)
2          tversky (set-based)
3   simple matching
4   braun_blanquet (set-based)
5        minkowski (vector-based)
6           ukkon (distribution-based)
7      sum_common (distribution-based)
8       distr_sim (distribution-based)
9   stringdot_utf8 (sequence-based)
10            pmi (special)
11       sim_emd (special)

Further to the similarity measures, melsim allows the user to specify which domain the
similarity should be calculated for. This is referred to as a "transformation" in melsim,
and all of the following transformations are implemented and functional:

Num:        Name:
1           pitch
2           int
3           fuzzy_int
4           parsons
5           pc
6           ioi_class
7           duration_class
8           int_X_ioi_class
9           implicit_harmonies

The following transformations are not currently functional in melsim:

Num:        Name:
1           ioi
2           phrase_segmentation

"""

from functools import cache, wraps
from types import SimpleNamespace
from pathlib import Path
from typing import Union, Tuple, List, Dict
import os
from multiprocessing import Pool, cpu_count
from itertools import combinations
from tqdm import tqdm

from tenacity import RetryError, Retrying, stop_after_attempt, wait_exponential
import numpy as np
from Feature_Set.import_mid import import_midi
import json
import subprocess

r_base_packages = ["base", "utils"]
r_cran_packages = [
    "tibble",
    "R6",
    "remotes",
    "dplyr",
    "magrittr",
    "proxy",
    "purrr",
    "purrrlyr",
    "tidyr",
    "yaml",
    "stringr",
    "emdist",
    "dtw",
    "ggplot2",
    "cba"
]
r_github_packages = ["melsim"]
github_repos = {
    "melsim": "sebsilas/melsim",
}

R = SimpleNamespace()


@cache
def load_melsim():
    check_python_package_installed("pandas")
    check_python_package_installed("rpy2")

    from rpy2.robjects import pandas2ri

    pandas2ri.activate()
    check_r_packages_installed()
    import_packages()


def requires_melsim(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        load_melsim()
        return func(*args, **kwargs)

    return wrapper


def check_r_packages_installed(install_missing: bool = False, n_retries: int = 3):
    from rpy2.robjects.packages import isinstalled

    for package in r_cran_packages + r_github_packages:
        if not isinstalled(package):
            if install_missing:
                try:
                    for attempt in Retrying(
                        stop=stop_after_attempt(n_retries),
                        wait=wait_exponential(multiplier=1, min=1, max=10),
                    ):
                        with attempt:
                            install_r_package(package)
                except RetryError as e:
                    raise RuntimeError(
                        f"Failed to install R package '{package}' after {n_retries} attempts. "
                        "See above for the traceback."
                    ) from e
            else:
                raise ImportError(
                    f"Package '{package}' is required but not installed. "
                    "You can install it by running: install_dependencies()"
                )


def install_r_package(package: str):
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import StrVector
    
    # Initialize R's utils package
    utils = importr("utils")
    
    # Set CRAN mirror
    utils.chooseCRANmirror(ind=1)  # Use the first mirror
    
    if package in r_cran_packages:
        print(f"Installing CRAN package '{package}'...")
        # Install with dependencies
        utils.install_packages(StrVector([package]), dependencies=True)
    elif package in r_github_packages:
        print(f"Installing GitHub package '{package}'...")
        remotes = importr("remotes")
        repo = github_repos[package]
        # Install with dependencies
        remotes.install_github(repo, upgrade="always", dependencies=True)
    else:
        raise ValueError(f"Unknown package type for '{package}'")


def install_dependencies():
    """Install all required R packages."""
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import StrVector
    
    # Initialize R's utils package
    utils = importr("utils")
    
    # Set CRAN mirror
    utils.chooseCRANmirror(ind=1)
    
    # Install all CRAN packages at once with dependencies
    print("Installing CRAN packages...")
    utils.install_packages(StrVector(r_cran_packages), dependencies=True)
    
    # Install itembankr from GitHub first
    print("Installing itembankr from GitHub...")
    remotes = importr("remotes")
    remotes.install_github("sebsilas/itembankr", upgrade="always", dependencies=True)
    
    # Install melsim from GitHub
    print("Installing melsim from GitHub...")
    for package in r_github_packages:
        repo = github_repos[package]
        print(f"Installing {package} from {repo}...")
        remotes.install_github(repo, upgrade="always", dependencies=True)
    
    print("All dependencies installed successfully!")


def import_packages():
    from rpy2.robjects.packages import importr

    all_packages = r_base_packages + r_cran_packages + r_github_packages
    for package in all_packages:
        setattr(R, package, importr(package))


def check_python_package_installed(package: str):
    try:
        __import__(package)
    except ImportError:
        raise ImportError(
            f"Package '{package}' is required but not installed. "
            f"Please install it using pip: pip install {package}"
        )


@requires_melsim
def get_similarity(
    melody1_pitches: np.ndarray,
    melody1_starts: np.ndarray,
    melody1_ends: np.ndarray,
    melody2_pitches: np.ndarray,
    melody2_starts: np.ndarray,
    melody2_ends: np.ndarray,
    method: str,
    transformation: str
) -> float:
    """Calculate similarity between two melodies using the specified method.

    Parameters
    ----------
    melody1_pitches : np.ndarray
        Array of MIDI pitch numbers for first melody
    melody1_starts : np.ndarray
        Array of start times for first melody notes
    melody1_ends : np.ndarray
        Array of end times for first melody notes
    melody2_pitches : np.ndarray
        Array of MIDI pitch numbers for second melody
    melody2_starts : np.ndarray
        Array of start times for second melody notes
    melody2_ends : np.ndarray
        Array of end times for second melody notes
    method : str
        Name of the similarity method to use from the list in the module docstring.
    transformation : str
        Name of the transformation to use from the list in the module docstring.

    Returns
    -------
    float
        Similarity value between the two melodies

    Examples
    --------
    >>> # Create two simple melodies
    >>> melody1_pitches = np.array([60, 62, 64, 65])  # C4, D4, E4, F4
    >>> melody1_starts = np.array([0.0, 1.0, 2.0, 3.0])
    >>> melody1_ends = np.array([0.9, 1.9, 2.9, 3.9])
    >>> melody2_pitches = np.array([60, 62, 64, 67])  # C4, D4, E4, G4
    >>> melody2_starts = np.array([0.0, 1.0, 2.0, 3.0])
    >>> melody2_ends = np.array([0.9, 1.9, 2.9, 3.9])
    >>> # Calculate similarity using Jaccard method
    >>> similarity = get_similarity(melody1_pitches, melody1_starts, melody1_ends,
    ...                           melody2_pitches, melody2_starts, melody2_ends,
    ...                           'Jaccard', 'pitch')
    """

    r_load_melody_arrays(melody1_pitches, melody1_starts, melody1_ends, "melody_1")
    r_load_melody_arrays(melody2_pitches, melody2_starts, melody2_ends, "melody_2")
    load_similarity_measure(method, transformation)
    return r_get_similarity("melody_1", "melody_2", method, transformation)


loaded_melodies = {}


@requires_melsim
def r_load_melody_arrays(pitches: np.ndarray, starts: np.ndarray, ends: np.ndarray, name: str):
    """Convert melody arrays to a format compatible with melsim R package.

    Args:
        pitches: Array of MIDI pitch numbers
        starts: Array of note start times
        ends: Array of note end times
        name: Name to assign the melody in R environment

    Returns:
        A melsim Melody object
    """
    import rpy2.robjects as ro
    from rpy2.robjects import FloatVector

    # Extract onset, pitch, duration for each note
    onsets = FloatVector(starts)
    pitches = FloatVector(pitches)
    durations = FloatVector(ends - starts)

    # Create R tibble using tibble::tibble()
    tibble = R.tibble.tibble(onset=onsets, pitch=pitches, duration=durations)

    ro.r.assign(f"{name}", ro.r("melody_factory$new")(mel_data=tibble))
    loaded_melodies[name] = (pitches, starts, ends)


@cache
def load_similarity_measure(method: str, transformation: str):
    import rpy2.robjects as ro

    valid_transformations = [
        "pitch",
        "int",
        "fuzzy_int",
        "parsons",
        "pc",
        "ioi_class",
        "duration_class",
        "int_X_ioi_class",
        "implicit_harmonies",
    ]

    # "ioi" and "phrase_segmentation" are not currently functional in melsim
    # but they will likely be added in the future

    if transformation not in valid_transformations:
        raise ValueError(f"Invalid transformation: {transformation}")

    ro.r.assign(
        f"{method}_sim",
        ro.r("sim_measure_factory$new")(
            name=method,
            full_name=method,
            transformation=transformation,
            parameters=ro.ListVector({}),
            sim_measure=method,
        ),
    )


@requires_melsim
def r_get_similarity(
    melody_1: str, melody_2: str, method: str, transformation: str
) -> float:
    """
    Use the melsim R package to get the similarity between two or more melodies.
    This version of get_similarity is designed to be used alongside r_load_melody_arrays.
    The user should call r_load_melody_arrays for each melody they wish to compare, and then
    call r_get_similarity for each pair of melodies. This is more efficient than
    calling get_similarity for each pair of melodies, as the melodies are only loaded once,
    and stored in memory for each subsequent call. Similarity measures are already cached,
    making this the faster way to calculate similarity between multiple melodies.

    Args:
        melody_1: Name of the first melody. This should have already been passed to R
        (see r_load_melody_arrays).
        melody_2: Name of the second melody. This should have already been passed to R.
        method: Name of the similarity method.
        transformation: Name of the transformation to use.

    Returns:
        The similarity value for each of the melody comparisons
    """
    import rpy2.robjects as ro

    # Load the similarity measure
    load_similarity_measure(method, transformation)

    return float(
        ro.r(f"{melody_1}$similarity")(ro.r(f"{melody_2}"), ro.r(f"{method}_sim")).rx2(
            "sim"
        )[0]
    )


def _convert_strings_to_tuples(d: Dict) -> Dict:
    """Convert string keys back to tuples where needed."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _convert_strings_to_tuples(v)
        else:
            result[k] = v
    return result


def load_midi_file(file_path: Union[str, Path]) -> Tuple[List[int], List[float], List[float]]:
    """Load MIDI file and extract melody attributes.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to MIDI file
        
    Returns
    -------
    Tuple[List[int], List[float], List[float]]
        Tuple of (pitches, start_times, end_times)
    """
    midi_data = import_midi(str(file_path))
    
    if midi_data is None:
        raise ValueError(f"Could not import MIDI file: {file_path}")
    
    return midi_data['pitches'], midi_data['starts'], midi_data['ends']


def _compute_similarity(args: Tuple) -> float:
    """Compute similarity between two melodies using R script.
    
    Parameters
    ----------
    args : Tuple
        Tuple containing (melody1_data, melody2_data, method, transformation)
        where melody_data is a tuple of (pitches, starts, ends)
        
    Returns
    -------
    float
        Similarity value
    """
    melody1_data, melody2_data, method, transformation = args
    
    # Convert lists to comma-separated strings
    pitches1_str = ",".join(map(str, melody1_data[0]))
    starts1_str = ",".join(map(str, melody1_data[1]))
    ends1_str = ",".join(map(str, melody1_data[2]))
    pitches2_str = ",".join(map(str, melody2_data[0]))
    starts2_str = ",".join(map(str, melody2_data[1]))
    ends2_str = ",".join(map(str, melody2_data[2]))
    
    # Get path to R script
    script_dir = Path(__file__).parent
    r_script = script_dir / "melsim.R"
    
    # Run R script
    try:
        result = subprocess.run(
            ["Rscript", str(r_script),
             pitches1_str, starts1_str, ends1_str,
             pitches2_str, starts2_str, ends2_str,
             method, transformation],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse JSON output and get first (and only) value
        return float(json.loads(result.stdout.strip())[0])
    except subprocess.CalledProcessError as e:
        print(f"Error running R script: {e.stderr}")
        raise


def _batch_compute_similarities(args_list: List[Tuple]) -> List[float]:
    """Compute similarities for a batch of melody pairs.
    
    Parameters
    ----------
    args_list : List[Tuple]
        List of argument tuples for _compute_similarity
        
    Returns
    -------
    List[float]
        List of similarity values
    """
    # Get path to R script
    script_dir = Path(__file__).parent
    r_script = script_dir / "melsim.R"
    
    # Prepare all arguments
    all_args = []
    for melody1_data, melody2_data, method, transformation in args_list:
        # Convert lists to comma-separated strings
        pitches1_str = ",".join(map(str, melody1_data[0]))
        starts1_str = ",".join(map(str, melody1_data[1]))
        ends1_str = ",".join(map(str, melody1_data[2]))
        pitches2_str = ",".join(map(str, melody2_data[0]))
        starts2_str = ",".join(map(str, melody2_data[1]))
        ends2_str = ",".join(map(str, melody2_data[2]))
        
        all_args.extend([
            pitches1_str, starts1_str, ends1_str,
            pitches2_str, starts2_str, ends2_str,
            method, transformation
        ])
    
    # Run R script with all arguments
    try:
        result = subprocess.run(
            ["Rscript", str(r_script)] + all_args,
            capture_output=True,
            text=True,
            check=True
        )
        # Parse JSON output and get all values
        return [float(x) for x in json.loads(result.stdout.strip())]
    except subprocess.CalledProcessError as e:
        print(f"Error running R script: {e.stderr}")
        raise


def get_similarity_from_midi(
    midi_path1: Union[str, Path],
    midi_path2: Union[str, Path] = None,
    method: Union[str, List[str]] = "Jaccard",
    transformation: Union[str, List[str]] = "pitch",
    output_file: Union[str, Path] = None,
    n_cores: int = None,
    batch_size: int = 500  # Process this many comparisons at once
) -> Union[float, Dict[Tuple[str, str, str, str], float]]:
    """Calculate similarity between MIDI files.
    
    If midi_path1 is a directory, performs pairwise comparisons between all MIDI files
    in the directory, ignoring midi_path2.
    
    You can provide a single method and transformation, or a list of methods and transformations.
    If you provide a list of methods and transformations, the function will return a dictionary
    mapping tuples of (file1, file2, method, transformation) to their similarity values.
    
    Parameters
    ----------
    midi_path1 : Union[str, Path]
        Path to first MIDI file or directory containing MIDI files
    midi_path2 : Union[str, Path], optional
        Path to second MIDI file. Ignored if midi_path1 is a directory
    method : Union[str, List[str]], default="Jaccard"
        Name of the similarity method(s) to use. Can be a single method or a list of methods.
    transformation : Union[str, List[str]], default="pitch"
        Name of the transformation(s) to use. Can be a single transformation or a list of transformations.
    output_file : Union[str, Path], optional
        If provided and doing pairwise comparisons, save results to this file.
        If no extension is provided, .json will be added.
    n_cores : int, optional
        Number of CPU cores to use for parallel processing. Defaults to all available cores.
    batch_size : int, default=100
        Number of comparisons to process in each batch
        
    Returns
    -------
    Union[float, Dict[Tuple[str, str, str, str], float]]
        If comparing two files, returns similarity value.
        If comparing all files in a directory, returns dictionary mapping tuples of
        (file1, file2, method, transformation) to their similarity values
    """
    # Convert single method/transformation to lists
    methods = [method] if isinstance(method, str) else method
    transformations = [transformation] if isinstance(transformation, str) else transformation
    
    midi_path1 = Path(midi_path1)
    
    # If midi_path1 is a directory, do pairwise comparisons
    if midi_path1.is_dir():
        midi_files = list(midi_path1.glob("*.mid"))
        
        if not midi_files:
            raise ValueError(f"No MIDI files found in {midi_path1}")
        
        # Load all melodies first
        print("Loading melodies...")
        melody_data = {}
        for file in tqdm(midi_files, desc="Loading MIDI files"):
            try:
                melody_data[file.name] = load_midi_file(file)
            except Exception as e:
                print(f"Warning: Could not load {file.name}: {str(e)}")
        
        if len(melody_data) < 2:
            raise ValueError("Need at least 2 valid MIDI files for comparison")
        
        # Prepare arguments for parallel processing
        print("Computing similarities...")
        args = []
        file_pairs = []
        for (name1, data1), (name2, data2) in combinations(melody_data.items(), 2):
            for m in methods:
                for t in transformations:
                    args.append((data1, data2, m, t))
                    file_pairs.append((name1, name2, m, t))
        
        # Process in batches
        similarities_list = []
        for i in tqdm(range(0, len(args), batch_size), desc="Processing batches"):
            batch = args[i:i + batch_size]
            similarities_list.extend(_batch_compute_similarities(batch))
        
        # Create dictionary of results
        similarities = {}
        for (name1, name2, m, t), sim in zip(file_pairs, similarities_list):
            similarities[(name1, name2, m, t)] = sim
        
        # Save to file if output file specified
        if output_file:
            print("Saving results...")
            import pandas as pd
            df = pd.DataFrame([
                {
                    "file1": f1,
                    "file2": f2,
                    "method": m,
                    "transformation": t,
                    "similarity": sim
                }
                for (f1, f2, m, t), sim in similarities.items()
            ])
            
            # Ensure output file has .json extension
            output_file = Path(output_file)
            if not output_file.suffix:
                output_file = output_file.with_suffix('.json')
            
            df.to_json(output_file, orient='records', indent=2)
            print(f"Results saved to {output_file}")
        
        return similarities
    
    # For single file comparison, only use first method and transformation
    if len(methods) > 1 or len(transformations) > 1:
        print("Warning: Multiple methods/transformations provided for two-file pairwise comparison. Using first method and transformation.")
    
    # Load MIDI files
    melody1_pitches, melody1_starts, melody1_ends = load_midi_file(midi_path1)
    melody2_pitches, melody2_starts, melody2_ends = load_midi_file(midi_path2)
    
    # Calculate similarity
    return _compute_similarity((
        (melody1_pitches, melody1_starts, melody1_ends),
        (melody2_pitches, melody2_starts, melody2_ends),
        methods[0], transformations[0]
    ))


@requires_melsim
def get_pairwise_similarities(
    midi_dir: Union[str, Path],
    method: str,
    transformation: str,
    output_file: Union[str, Path] = None
) -> Dict[Tuple[str, str], float]:
    """Calculate pairwise similarities between all MIDI files in a directory.
    
    Parameters
    ----------
    midi_dir : Union[str, Path]
        Path to directory containing MIDI files
    method : str
        Name of the similarity method to use
    transformation : str
        Name of the transformation to use
    output_file : Union[str, Path], optional
        If provided, save results to this .json file
        
    Returns
    -------
    Dict[Tuple[str, str], float]
        Dictionary mapping pairs of filenames to their similarity values
        
    Examples
    --------
    >>> similarities = get_pairwise_similarities(
    ...     "path/to/midi/directory",
    ...     "Jaccard",
    ...     "pitch",
    ...     "similarities.json"
    ... )
    """
    midi_dir = Path(midi_dir)
    midi_files = list(midi_dir.glob("*.mid"))
    
    if not midi_files:
        raise ValueError(f"No MIDI files found in {midi_dir}")
    
    similarities = {}
    
    # Compare each pair of files
    for i, file1 in enumerate(midi_files):
        for file2 in midi_files[i+1:]:  # Only compare each pair once
            try:
                similarity = get_similarity_from_midi(
                    file1, file2,
                    method=method,
                    transformation=transformation
                )
                similarities[(file1.name, file2.name)] = similarity
            except Exception as e:
                print(f"Warning: Could not compare {file1.name} and {file2.name}: {str(e)}")
    
    # Save to json if output file specified
    if output_file:
        import pandas as pd
        df = pd.DataFrame([
            {"file1": f1, "file2": f2, "similarity": sim}
            for (f1, f2), sim in similarities.items()
        ])
        df.to_json(output_file, index=False)
    
    return similarities