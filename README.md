This repo collects all the work I am currently conducting for my PhD. It is organised as follows:
- `Essen_Analysis` houses files used to test my melodic features software
  
- `Feature_Set` provides access to this melodic features software: run `pip install -e .` in the clone directory

- `PhDMelodySet.egg-info` helps with setup

- `RotationForest` contains a copy of a Python package for statistical analysis involved in the modelling work

- `modelling_paper` contains all scripts relevant to the Harvard MDT modelling I've done so far

- `requirements.txt` is used to install dependencies for the feature set

- `setup.py` builds the feature set as a Python module

This is very much a living repository and I can only apologise for the disorder - I will clean it up again soon!  

========================================= 

**Feature Set**

To use the feature set, first clone the repo to a destination of your choosing.
Navigate to this directory and create a new virtual environment, this can be achieved using:  

`python3 -m venv venv`  

`source venv/bin/activate`  

(this assumes you are using `zsh` as your terminal)  


Once in your venv, dependencies can be installed using `pip install -r "requirements.txt"`  
This will automatically gather the required Python packages needed to run the feature set.  

We can then build the feature set as a module by running `pip install -e .`  
This will allow us to import the feature set as a proper Python module.  

=========================================  

**Example Usage**

Once built, the feature set is very easy to use.  
We simply supply a directory containing MIDI files, and specify an output file name.

Where a corpus is not required, we can compute all the other features in a simple script:

```py
from Feature_Set.features import get_all_features

# we must use the 'if name main' convention here - omitting this guard will result in a circular import

if __name__ == "__main__":
    get_all_features("path_to_midi_file_directory", "name_of_output_file")
```

This will produce `name_of_output_file.csv`, containing a row for every melody and every melodic feature calculable.

If we wish to produce corpus features, we have to do some more work first. We build a corpus dictionary using `corpus.py` as follows:

```py
from Feature_Set.corpus import make_corpus_stats
from Feature_Set.features import get_all_features

# using the same name is main guard

if __name__ == "__main__":
    make_corpus_stats("path_to_midi_file_directory", "name_of_output_dict")

    # We can then use the produced `name_of_output_dict.json` file as the third argument in our `get_all_features` function

    get_all_features("path_to_midi_file_directory", "name_of_output_file", "name_of_output_dict.json")
```

It's as simple as that!


=========================================  
**A Note on Melsim**

I am currently aware of issues with using `melsim.py` - this seems to be linked to the version of R that the user has stored in their `$PATH` variable.  

My best guess is that `rpy2` isn't currently up-to-date with R 4.5.0 - will introduce a robust fix in time, but for now I encourage the user to use R 4.4.x if they wish to use this wrapper.  

Sorry for the inconvenience!
