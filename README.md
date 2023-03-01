# FastFuzzyMatching

FuzzyMatcher is a Python library for fuzzy string matching using the fuzzywuzzy package. It provides an easy-to-use interface for matching two pandas dataframes, with the ability to customize the matching rules and scoring criteria. The matching can be performed using either a simple matching algorithm or a deep matching algorithm.

## Installation

To get FuzzyMatcher please clone repository via:

```bash
git clone "https://github.com/burik193/FastFuzzyMatching.git"
```

## Usage

```bash
from fuzzyMatcher.matcher import FuzzyMatcher
import os
from pathlib import Path
# from IPython.display import display

# Here is an example of how to use FuzzyMatcher

if __name__ == '__main__':
    root = Path(os.getcwd())
    lhs = root.joinpath('path_to_left_df') # Dataset 87816 x 44 - 10.777 distinct names
    rhs = root.joinpath('path_to_right_df')
    M = FuzzyMatcher(lhs=lhs, rhs=rhs, left_on='col_name_on_lhs', right_on='col_name_on_rhs', verbose=True)
    match = M.merge(deep=True, max_matches=3)
    #display(match)
    match.to_excel('produced_match.xlsx')

    match = M.make_scoring(rules={'categorical_column_name': [('value_1', 1.5), ('value_2', 1)], 'value_3': None})

    match.to_excel('produced_match_scored.xlsx')
```

## Features
- Simple string matching algorithm
- Deep string matching algorithm
- Customizable matching rules
- Customizable scoring rules
- Ability to match and merge large dataframes using dask

## Contributing
We welcome contributions to FuzzyMatcher. If you encounter a bug, have a feature request, or would like to contribute a new feature or enhancement, please open an issue or submit a pull request.

## License
FastFuzzyMatching is released under the MIT License.
