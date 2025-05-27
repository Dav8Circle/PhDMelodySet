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

from tenacity import RetryError, Retrying, stop_after_attempt, wait_exponential
from typing import List, Union
import numpy as np

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