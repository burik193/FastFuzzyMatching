# FuzzyMatcher

FuzzyMatcher is a Python library for fuzzy string matching using the fuzzywuzzy package. It provides an easy-to-use interface for matching two pandas dataframes, with the ability to customize the matching rules and scoring criteria. The matching can be performed using either a simple matching algorithm or a deep matching algorithm.

## Installation

FuzzyMatcher can be installed via pip:

```bash
pip install fuzzymatcher

## Usage

from fuzzymatcher import FuzzyMatcher

# Initialize the FuzzyMatcher object
matcher = FuzzyMatcher()

# Load the left and right dataframes
matcher.load_left(left_df)
matcher.load_right(right_df)

# Set the left and right join keys
matcher.set_left_on('left_key')
matcher.set_right_on('right_key')

# Match the dataframes
matcher.match()

# Score the matches using custom rules
matcher.make_scoring({'column1': [('value1', 0.5), ('value2', 0.8)], 'column2': None})

# Merge the dataframes based on the matches
matcher.merge()


## Features
- Simple string matching algorithm
- Deep string matching algorithm
- Customizable matching rules
- Customizable scoring rules
- Ability to match and merge large dataframes using dask

## Contributing
We welcome contributions to FuzzyMatcher. If you encounter a bug, have a feature request, or would like to contribute a new feature or enhancement, please open an issue or submit a pull request.

License
FuzzyMatcher is released under the MIT License.
